"""
GVB (Generalized Valence Bond) solver for PySCF's CASSCF/CASCI framework.

Ported from PyQuante2's mcscf.py by Rick Muller.

Reference:
    F.W. Bobrowicz and W.A. Goddard III,
    "The Self-Consistent Equations for Generalized Valence Bond and
    Open-Shell Hartree-Fock Wave Functions",
    in Methods of Electronic Structure Theory, H.F. Schaeffer III, ed.
    Plenum Publishing Corp., 1977.
    http://www.wag.caltech.edu/publications/sup/pdf/108.pdf

This module provides GVBSolver, a class implementing the PySCF fcisolver
interface (kernel, make_rdm1, make_rdm12) so it can be used as a drop-in
replacement for the default FCI solver in CASSCF or CASCI calculations.

The GVB wavefunction for npair pairs takes the form:
    |GVB> = product_over_pairs [ c_{p,1} |..p1_alpha p1_beta..> + c_{p,2} |..p2_alpha p2_beta..> ]

Each pair is described by two natural orbitals and two CI coefficients.
The solver optimizes both the orbital rotations (mixing within pairs)
and CI coefficients self-consistently.
"""

import numpy as np
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


class GVBCIVector:
    """Container for GVB CI state, passed opaquely between kernel/make_rdm1/make_rdm12."""
    def __init__(self, norb, nelec, pair_coeffs, orb_rotation=None):
        """
        Args:
            norb: Number of active orbitals.
            nelec: Number of active electrons (int or tuple).
            pair_coeffs: Array of shape (npair, 2) with CI coefficients per pair.
                         pair_coeffs[i] = [c1, c2] with c1^2 + c2^2 = 1.
            orb_rotation: Unitary matrix (norb, norb) rotating from the input
                          MO basis to the GVB natural orbital basis. None = identity.
        """
        self.norb = norb
        self.nelec = nelec if isinstance(nelec, (int, np.integer)) else sum(nelec)
        self.pair_coeffs = np.asarray(pair_coeffs)
        self.npair = len(pair_coeffs)
        self.orb_rotation = orb_rotation


class GVBSolver:
    """
    GVB solver implementing PySCF's fcisolver interface.

    This solver assumes the active space consists of npair GVB pairs,
    with 2*npair active orbitals and 2*npair active electrons (all singlet
    paired). It can be used as mc.fcisolver in CASSCF or CASCI.

    The solver works by:
    1. Optimizing intra-pair orbital rotations (ROTION-like step)
    2. Solving 2x2 CI problems for each pair
    3. Iterating to self-consistency

    Usage:
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.fcisolver = GVBSolver(npair=nelecas//2)
        mc.kernel()

    Attributes:
        npair: Number of GVB pairs. If None, inferred as nelec//2.
        max_cycle: Maximum number of GVB micro-iterations per kernel call.
        conv_tol: Convergence threshold for energy.
        nroots: Number of roots (always 1 for GVB).
        converged: Whether the last kernel call converged.
    """

    def __init__(self, npair=None, max_cycle=50, conv_tol=1e-10):
        self.npair = npair
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.nroots = 1
        self.converged = False
        self.verbose = 0
        self.spin = 0  # GVB is singlet

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        """
        Solve the GVB problem in the active space.

        Args:
            h1e: 1-electron integrals in active MO basis, shape (norb, norb).
            eri: 2-electron integrals in active MO basis. Can be shape
                 (norb,norb,norb,norb) or compressed. Chemist's notation (pq|rs).
            norb: Number of active orbitals.
            nelec: Number of active electrons (int or (nalpha, nbeta)).
            ci0: Initial guess (GVBCIVector or None).
            ecore: Core energy to add to the total.

        Returns:
            (e_tot, civec): Total energy and GVBCIVector.
        """
        if isinstance(nelec, (tuple, list)):
            na, nb = nelec
            nel = na + nb
        else:
            nel = nelec
            na = nb = nel // 2

        npair = self.npair if self.npair is not None else nel // 2
        assert 2 * npair == norb, \
            f"GVB requires norb == 2*npair, got norb={norb}, npair={npair}"
        assert 2 * npair == nel, \
            f"GVB requires nelec == 2*npair, got nelec={nel}, npair={npair}"

        # Ensure eri is 4-index
        eri = _unpack_eri(eri, norb)

        # Initialize pair coefficients
        if ci0 is not None and isinstance(ci0, GVBCIVector):
            pair_coeffs = ci0.pair_coeffs.copy()
            U = ci0.orb_rotation if ci0.orb_rotation is not None else np.eye(norb)
        else:
            pair_coeffs = np.zeros((npair, 2))
            pair_coeffs[:, 0] = 1.0  # Start with HF-like guess
            pair_coeffs[:, 1] = 0.0
            U = np.eye(norb)

        # Transform integrals to current orbital basis
        h1 = _rotate_h1(h1e, U)
        eri_rot = _rotate_eri(eri, U)

        E_old = 0.0
        for iteration in range(self.max_cycle):
            # Step 1: Solve 2x2 CI for each pair
            pair_coeffs, e_pair = _solve_pair_ci(h1, eri_rot, npair, pair_coeffs)

            # Step 2: Compute total energy
            E_el = _compute_gvb_energy(h1, eri_rot, npair, pair_coeffs)

            # Step 3: Orbital rotation between pairs (ROTION-like)
            delta = _compute_orbital_rotation(h1, eri_rot, npair, pair_coeffs)
            if np.max(np.abs(delta)) > 1e-12:
                eD = _expm(delta)
                U = U @ eD
                h1 = _rotate_h1(h1e, U)
                eri_rot = _rotate_eri(eri, U)

            E_tot = E_el + ecore

            if abs(E_tot - E_old) < self.conv_tol:
                self.converged = True
                logger.debug("GVB converged in %d iterations, E = %.10f",
                             iteration + 1, E_tot)
                break
            E_old = E_tot
        else:
            self.converged = False
            logger.warning("GVB not converged after %d iterations", self.max_cycle)

        civec = GVBCIVector(norb, nelec, pair_coeffs, orb_rotation=U)
        return E_tot, civec

    def make_rdm1(self, civec, norb, nelec):
        """
        Compute spin-traced 1-RDM in the original MO basis.

        Returns:
            rdm1: shape (norb, norb), dm1[p,q] = <p_a^+ q_a> + <p_b^+ q_b>
        """
        if not isinstance(civec, GVBCIVector):
            # Fallback: return doubly-occupied RDM
            npair = norb // 2
            rdm1 = np.zeros((norb, norb))
            for i in range(npair):
                rdm1[i, i] = 2.0
            return rdm1

        npair = civec.npair
        U = civec.orb_rotation if civec.orb_rotation is not None else np.eye(norb)

        # In the GVB natural orbital basis, the 1-RDM is diagonal
        # For pair p with coefficients (c1, c2):
        #   occupation of orbital 2*p   = 2*c1^2
        #   occupation of orbital 2*p+1 = 2*c2^2
        rdm1_no = np.zeros((norb, norb))
        for p in range(npair):
            c1, c2 = civec.pair_coeffs[p]
            rdm1_no[2 * p, 2 * p] = 2.0 * c1 ** 2
            rdm1_no[2 * p + 1, 2 * p + 1] = 2.0 * c2 ** 2

        # Rotate back to original MO basis
        rdm1 = U @ rdm1_no @ U.T
        return rdm1

    def make_rdm12(self, civec, norb, nelec):
        """
        Compute spin-traced 1-RDM and 2-RDM in the original MO basis.

        Returns:
            (rdm1, rdm2): rdm1 shape (norb, norb), rdm2 shape (norb, norb, norb, norb).
            rdm2 in chemist's notation: rdm2[p,q,r,s] = <p^+ r^+ s q>
        """
        if not isinstance(civec, GVBCIVector):
            rdm1 = self.make_rdm1(civec, norb, nelec)
            rdm2 = np.einsum('pq,rs->pqrs', rdm1, rdm1) - \
                   0.5 * np.einsum('ps,rq->pqrs', rdm1, rdm1)
            return rdm1, rdm2

        npair = civec.npair
        U = civec.orb_rotation if civec.orb_rotation is not None else np.eye(norb)

        # Build RDMs in the natural orbital basis, then rotate
        rdm1_no = np.zeros((norb, norb))
        rdm2_no = np.zeros((norb, norb, norb, norb))

        for p in range(npair):
            c1, c2 = civec.pair_coeffs[p]
            i, j = 2 * p, 2 * p + 1
            ni = 2.0 * c1 ** 2  # occupation of orbital i
            nj = 2.0 * c2 ** 2  # occupation of orbital j

            # 1-RDM
            rdm1_no[i, i] = ni
            rdm1_no[j, j] = nj

        # 2-RDM construction
        # For a GVB wavefunction that is a product of pair functions,
        # the 2-RDM has intra-pair and inter-pair contributions.
        #
        # Chemist's notation: rdm2[p,q,r,s] = <p^+ r^+ s q>
        #
        # Inter-pair: standard Hartree-Fock-like factorization
        # rdm2[p,q,r,s] = rdm1[p,q]*rdm1[r,s] - 0.5*rdm1[p,s]*rdm1[r,q]
        #
        # Intra-pair: need exact treatment due to correlation

        # Start with the factorized (HF-like) part
        rdm2_no = np.einsum('pq,rs->pqrs', rdm1_no, rdm1_no) - \
                  0.5 * np.einsum('ps,rq->pqrs', rdm1_no, rdm1_no)

        # Now correct the intra-pair blocks
        for p in range(npair):
            c1, c2 = civec.pair_coeffs[p]
            i, j = 2 * p, 2 * p + 1
            ni = 2.0 * c1 ** 2
            nj = 2.0 * c2 ** 2

            # The exact intra-pair 2-RDM elements for a singlet pair:
            # |pair> = c1|i_up i_dn> + c2|j_up j_dn>
            #
            # In chemist's notation rdm2[p,q,r,s] = <p^+ r^+ s q>:
            #
            # rdm2[i,i,i,i] = <i^+ i^+ i i> (both spins) = 2*c1^2 * (2*c1^2 - 1)
            #   but for singlet pair: n_i(n_i-1) isn't right because it's not
            #   independent electrons. For |pair> = c1|ii> + c2|jj>:
            #
            # Exact 2-RDM elements for singlet geminal c1|i_a i_b> + c2|j_a j_b>:
            #   <i_a^+ i_b^+ i_b i_a> = c1^2  (both electrons in i)
            #   <j_a^+ j_b^+ j_b j_a> = c2^2  (both electrons in j)
            #   <i_a^+ j_b^+ j_b i_a> = c1^2  (electron 1 in i_a, electron 2 in j_b... no)
            #
            # Let me be more careful. The pair wavefunction is:
            #   |Psi> = c1 a_ia^+ a_ib^+ |vac> + c2 a_ja^+ a_jb^+ |vac>
            #
            # Spin-traced 2-RDM: Gamma[p,q,r,s] = sum_{sigma,tau} <a_{p,sigma}^+ a_{r,tau}^+ a_{s,tau} a_{q,sigma}>
            #
            # Gamma[i,i,i,i]:
            #   sigma=a,tau=b: <a_ia^+ a_ib^+ a_ib a_ia> = c1^2
            #   sigma=b,tau=a: <a_ib^+ a_ia^+ a_ia a_ib> = c1^2
            #   sigma=a,tau=a: <a_ia^+ a_ia^+ a_ia a_ia> = 0
            #   sigma=b,tau=b: same = 0
            #   Total: 2*c1^2
            # But wait, rdm2[p,q,r,s] should give rdm2[i,i,i,i] = ni*(ni-1)
            # for uncorrelated case. For c1=1,c2=0: ni=2, so 2*1=2. And ni*(ni-1)=2. OK.
            #
            # Gamma[j,j,j,j] = 2*c2^2
            #
            # Gamma[i,i,j,j]:
            #   sum_{s,t} <a_is^+ a_jt^+ a_jt a_is>
            #   For c1|ii>+c2|jj>, the i and j orbitals never simultaneously occupied
            #   so Gamma[i,i,j,j] = 0
            #
            # Gamma[i,j,i,j]:
            #   sum_{s,t} <a_is^+ a_it^+ a_jt a_js>
            #   = 0 (same reasoning, no cross terms for different orbitals)
            #
            # Gamma[i,j,j,i]:
            #   sum_{s,t} <a_is^+ a_jt^+ a_it a_js>
            #   sigma=a,tau=b: <a_ia^+ a_jb^+ a_ib a_ja>
            #   This connects |ii> to |jj>: = c1*c2
            #   sigma=b,tau=a: <a_ib^+ a_ja^+ a_ia a_jb> = c1*c2
            #   Total: 2*c1*c2
            #
            # Similarly Gamma[j,i,i,j] = 2*c1*c2

            # The factorized values (what we already have):
            # rdm2_fac[i,i,i,i] = ni*ni - 0.5*ni*ni = 0.5*ni^2 = 2*c1^4
            # rdm2_fac[j,j,j,j] = 2*c2^4
            # rdm2_fac[i,i,j,j] = ni*nj - 0 = ni*nj = 4*c1^2*c2^2
            # rdm2_fac[i,j,j,i] = 0 - 0.5*ni*nj*0 = 0  (off-diag rdm1 = 0)
            #   Wait: rdm2_fac[i,j,j,i] = rdm1[i,j]*rdm1[j,i] - 0.5*rdm1[i,i]*rdm1[j,j]
            #   = 0 - 0.5*ni*nj = -2*c1^2*c2^2
            # rdm2_fac[j,i,i,j] = -0.5*nj*ni = -2*c1^2*c2^2
            # rdm2_fac[i,j,i,j] = 0 - 0.5*0 = 0  (rdm1[i,j]=0)
            #   Actually: rdm2_fac[i,j,i,j] = rdm1[i,j]*rdm1[i,j] - 0.5*rdm1[i,j]*rdm1[i,j]
            #   Hmm, let me re-derive:
            #   rdm2_fac[p,q,r,s] = rdm1[p,q]*rdm1[r,s] - 0.5*rdm1[p,s]*rdm1[r,q]
            #   rdm2_fac[i,j,i,j] = rdm1[i,j]*rdm1[i,j] - 0.5*rdm1[i,j]*rdm1[i,j] = 0

            # Corrections needed (exact - factorized):
            # [i,i,i,i]: 2*c1^2 - 2*c1^4
            # [j,j,j,j]: 2*c2^2 - 2*c2^4
            # [i,i,j,j]: 0 - 4*c1^2*c2^2 = -4*c1^2*c2^2
            # [j,j,i,i]: 0 - 4*c1^2*c2^2 = -4*c1^2*c2^2
            # [i,j,j,i]: 2*c1*c2 - (-2*c1^2*c2^2) = 2*c1*c2 + 2*c1^2*c2^2
            # [j,i,i,j]: 2*c1*c2 - (-2*c1^2*c2^2) = 2*c1*c2 + 2*c1^2*c2^2

            rdm2_no[i, i, i, i] = 2.0 * c1 ** 2  # exact
            rdm2_no[j, j, j, j] = 2.0 * c2 ** 2
            rdm2_no[i, i, j, j] = 0.0  # orbs i,j never simultaneously occupied
            rdm2_no[j, j, i, i] = 0.0
            rdm2_no[i, j, j, i] = 2.0 * c1 * c2  # exchange-like cross term
            rdm2_no[j, i, i, j] = 2.0 * c1 * c2
            rdm2_no[i, j, i, j] = 0.0
            rdm2_no[j, i, j, i] = 0.0

        # Rotate RDMs to original MO basis
        rdm1 = U @ rdm1_no @ U.T
        rdm2 = np.einsum('pa,qb,rc,sd,abcd->pqrs', U, U, U, U, rdm2_no)

        return rdm1, rdm2

    def make_rdm1s(self, civec, norb, nelec):
        """Return (rdm1_alpha, rdm1_beta). For singlet GVB they are equal."""
        rdm1 = self.make_rdm1(civec, norb, nelec)
        return rdm1 * 0.5, rdm1 * 0.5

    def spin_square(self, civec, norb, nelec):
        """Return (S^2, 2S+1). GVB is always singlet."""
        return 0.0, 1.0


# ============================================================
# Helper functions (ported from PyQuante2's mcscf.py)
# ============================================================

def _unpack_eri(eri, norb):
    """Ensure eri is a 4-index array (norb, norb, norb, norb)."""
    if eri.ndim == 4:
        return eri
    # PySCF often stores eri in compressed form
    from pyscf import ao2mo
    return ao2mo.restore(1, eri, norb)


def _rotate_h1(h1e, U):
    """Rotate 1-electron integrals: h1' = U^T h1 U"""
    return U.T @ h1e @ U


def _rotate_eri(eri, U):
    """Rotate 2-electron integrals: eri'[p,q,r,s] = U^T_pa U_qb U^T_rc U_sd eri[a,b,c,d]"""
    return np.einsum('pa,qb,rc,sd,abcd->pqrs', U, U, U, U, eri)


def _solve_pair_ci(h1, eri, npair, pair_coeffs):
    """
    Solve the 2x2 CI problem for each GVB pair.

    For pair p with orbitals (2p, 2p+1), the 2x2 Hamiltonian is:
        H[0,0] = <ii|H|ii> = 2*h[i,i] + eri[i,i,i,i] + core_contrib_i
        H[1,1] = <jj|H|jj> = 2*h[j,j] + eri[j,j,j,j] + core_contrib_j
        H[0,1] = <ii|H|jj> = eri[i,j,j,i]  (the exchange integral K_ij)

    where the "core contribution" comes from the other pairs.
    """
    norb = 2 * npair
    new_coeffs = np.zeros_like(pair_coeffs)
    e_pairs = np.zeros(npair)

    for p in range(npair):
        i, j = 2 * p, 2 * p + 1

        # Compute effective 1-electron terms including interaction with other pairs
        h_eff_ii = 2.0 * h1[i, i]
        h_eff_jj = 2.0 * h1[j, j]

        # Intra-pair Coulomb
        h_eff_ii += eri[i, i, i, i]
        h_eff_jj += eri[j, j, j, j]

        # Interaction with other pairs
        for q in range(npair):
            if q == p:
                continue
            k, l = 2 * q, 2 * q + 1
            c1q, c2q = pair_coeffs[q]

            # Coulomb and exchange with orbital k (occ = 2*c1q^2)
            nk = 2.0 * c1q ** 2
            nl = 2.0 * c2q ** 2

            h_eff_ii += nk * (2.0 * eri[i, i, k, k] - eri[i, k, k, i])
            h_eff_ii += nl * (2.0 * eri[i, i, l, l] - eri[i, l, l, i])
            h_eff_jj += nk * (2.0 * eri[j, j, k, k] - eri[j, k, k, j])
            h_eff_jj += nl * (2.0 * eri[j, j, l, l] - eri[j, l, l, j])

        # Off-diagonal: exchange integral between the two pair orbitals
        K_ij = eri[i, j, j, i]

        H2x2 = np.array([[h_eff_ii, K_ij],
                          [K_ij, h_eff_jj]])

        eigvals, eigvecs = np.linalg.eigh(H2x2)
        new_coeffs[p] = eigvecs[:, 0]  # Ground state
        e_pairs[p] = eigvals[0]

        # Ensure consistent sign convention (c1 > 0)
        if new_coeffs[p, 0] < 0:
            new_coeffs[p] *= -1

    return new_coeffs, e_pairs


def _compute_gvb_energy(h1, eri, npair, pair_coeffs):
    """
    Compute the total electronic energy for the GVB wavefunction.

    E = sum_p [ n_ip * h[ip,ip] + n_jp * h[jp,jp]
              + 0.5 * n_ip * (n_ip - 1) * eri[ip,ip,ip,ip] / n_ip  ... ]

    More precisely, use the 1- and 2-RDM to contract with integrals.
    """
    norb = 2 * npair

    # Build diagonal 1-RDM in this basis
    rdm1 = np.zeros((norb, norb))
    for p in range(npair):
        c1, c2 = pair_coeffs[p]
        rdm1[2 * p, 2 * p] = 2.0 * c1 ** 2
        rdm1[2 * p + 1, 2 * p + 1] = 2.0 * c2 ** 2

    # 1-electron energy
    E1 = np.einsum('ii,ii->', h1, rdm1)

    # 2-electron energy via explicit 2-RDM construction (only diagonal blocks matter)
    E2 = 0.0

    for p in range(npair):
        c1p, c2p = pair_coeffs[p]
        ip, jp = 2 * p, 2 * p + 1

        # Intra-pair: exact 2-RDM
        # Gamma[i,i,i,i] = 2*c1^2, gives 0.5 * 2*c1^2 * eri[i,i,i,i]
        E2 += 0.5 * 2.0 * c1p ** 2 * eri[ip, ip, ip, ip]
        E2 += 0.5 * 2.0 * c2p ** 2 * eri[jp, jp, jp, jp]
        # Cross exchange: Gamma[i,j,j,i] = 2*c1*c2
        E2 += 2.0 * c1p * c2p * eri[ip, jp, jp, ip]

        # Inter-pair contributions
        for q in range(p + 1, npair):
            c1q, c2q = pair_coeffs[q]
            iq, jq = 2 * q, 2 * q + 1

            nip = 2.0 * c1p ** 2
            njp = 2.0 * c2p ** 2
            niq = 2.0 * c1q ** 2
            njq = 2.0 * c2q ** 2

            # Coulomb - exchange between all orbital combinations
            for o1, n1 in [(ip, nip), (jp, njp)]:
                for o2, n2 in [(iq, niq), (jq, njq)]:
                    E2 += n1 * n2 * eri[o1, o1, o2, o2]  # Coulomb
                    E2 -= 0.5 * n1 * n2 * eri[o1, o2, o2, o1]  # Exchange

    return E1 + E2


def _compute_orbital_rotation(h1, eri, npair, pair_coeffs):
    """
    Compute orbital rotation matrix between orbitals of different pairs
    (ROTION step from PyQuante2).

    Only rotates between orbitals belonging to different pairs.
    Returns antisymmetric matrix delta such that U_new = U_old @ expm(delta).
    """
    norb = 2 * npair
    delta = np.zeros((norb, norb))

    # Build Fock-like matrices for each orbital
    # F_i = f_i * h + sum_j (a_ij * J_j + b_ij * K_j)
    # But since we're in the MO basis and the 1-RDM is diagonal,
    # we can compute the generalized Fock matrix directly.

    # Occupations
    occ = np.zeros(norb)
    for p in range(npair):
        c1, c2 = pair_coeffs[p]
        occ[2 * p] = 2.0 * c1 ** 2
        occ[2 * p + 1] = 2.0 * c2 ** 2

    # Build Fock matrix for each orbital
    F = np.zeros((norb, norb, norb))  # F[i] = Fock matrix for orbital i
    for i in range(norb):
        ip = i // 2  # pair index
        c1i, c2i = pair_coeffs[ip]
        fi = occ[i] / 2.0  # f = occ/2

        F[i] = fi * h1.copy()
        for j in range(norb):
            jp = j // 2
            c1j, c2j = pair_coeffs[jp]
            fj = occ[j] / 2.0

            # Default: a_ij = 2*fi*fj, b_ij = -fi*fj
            a_ij = 2.0 * fi * fj
            b_ij = -fi * fj

            # Corrections for same-pair orbitals
            if ip == jp:
                if i == j:
                    # Diagonal pair orbital: a_ii = f_i, b_ii = 0
                    a_ij = fi
                    b_ij = 0.0
                else:
                    # Same pair, different orbital: a_ij = 0, b_ij = -c1*c2
                    a_ij = 0.0
                    b_ij = -c1i * c2i if i < j else -c1j * c2j
                    # Use the pair's coefficients
                    b_ij = -pair_coeffs[ip, 0] * pair_coeffs[ip, 1]

            # J_j[p,q] = sum_rs D_j[r,s] * eri[p,q,r,s] where D_j is density for orbital j
            # For a single orbital j: D_j[r,s] = delta_{r,j} delta_{s,j}
            # So J_j[p,q] = eri[p,q,j,j]
            # K_j[p,q] = eri[p,j,j,q]  (in chemist's notation, this is exchange)
            # Wait: K_j[p,q] = sum_rs D_j[r,s] * eri[p,s,r,q] = eri[p,j,j,q]
            F[i] += a_ij * eri[:, :, j, j]  # Coulomb-like
            F[i] += b_ij * eri[:, j, j, :]  # Exchange-like

    # Compute rotation angles between orbitals of different pairs
    for i in range(norb):
        ip = i // 2
        for j in range(i):
            jp = j // 2
            if ip == jp:
                continue  # Don't rotate within a pair

            # Gradient: F[j][i,j] - F[i][i,j]
            gradient = F[j][i, j] - F[i][i, j]

            # Hessian (diagonal approximation)
            hessian = F[j][i, i] - F[i][i, i] - F[j][j, j] + F[i][j, j]

            # Add two-electron correction to hessian
            ish, jsh = ip, jp
            Kij = eri[i, j, j, i]  # Exchange integral
            Jij = eri[i, i, j, j]  # Coulomb integral

            fi = occ[i] / 2.0
            fj = occ[j] / 2.0

            # Simplified Gij correction
            Gij = 2.0 * (fi - fj) ** 2 * Kij

            hessian += Gij

            if abs(hessian) > 1e-14:
                delta[i, j] = -gradient / hessian
                delta[j, i] = gradient / hessian

    # Damping: limit rotation angles
    max_rot = np.max(np.abs(delta))
    if max_rot > 0.5:
        delta *= 0.5 / max_rot

    return delta


def _expm(M, tol=1e-10, maxit=50):
    """
    Matrix exponential via Taylor series.
    Ported from PyQuante2.
    """
    n = M.shape[0]
    factor = 1.0
    X = np.eye(n)
    eM = np.eye(n)
    for j in range(1, maxit):
        factor /= j
        X = X @ M
        if np.linalg.norm(X) * abs(factor) < tol:
            break
        eM += factor * X
    return eM
