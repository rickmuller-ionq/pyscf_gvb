"""
Tests for the GVB solver as a PySCF fcisolver.

Tests include:
1. H2 with STO-3G: GVB(1 pair) should give lower energy than RHF
2. RDM consistency checks
3. CASCI with GVB solver vs standard FCI
"""

import numpy as np
import pytest
from pyscf import gto, scf, mcscf, fci
from pyscf_gvb import GVBSolver


def test_h2_sto3g_casci():
    """H2/STO-3G: GVB(1) via CASCI should recover correlation energy."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    assert mf.converged

    # CASCI with 2 orbitals, 2 electrons
    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver = GVBSolver(npair=1)
    mc.verbose = 0
    mc.kernel()

    # RHF energy for H2/STO-3G is about -1.1171
    # FCI energy for H2/STO-3G is about -1.1373
    # GVB(1) should match FCI for 2 electrons in 2 orbitals
    assert mc.e_tot < mf.e_tot, "GVB energy should be lower than RHF"
    print(f"RHF energy:  {mf.e_tot:.8f}")
    print(f"GVB energy:  {mc.e_tot:.8f}")
    print(f"Correlation: {mc.e_tot - mf.e_tot:.8f}")


def test_h2_sto3g_casscf():
    """H2/STO-3G: GVB(1) via CASSCF should match CASCI for minimal basis."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()

    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fcisolver = GVBSolver(npair=1)
    mc.verbose = 0
    mc.kernel()

    # Compare with FCI
    mc_fci = mcscf.CASCI(mf, 2, 2)
    mc_fci.verbose = 0
    mc_fci.kernel()

    print(f"GVB-CASSCF energy: {mc.e_tot:.8f}")
    print(f"FCI-CASCI energy:  {mc_fci.e_tot:.8f}")

    # For H2/STO-3G (minimal basis, 2 orbs, 2 elec), GVB(1) = FCI
    assert abs(mc.e_tot - mc_fci.e_tot) < 1e-6, \
        f"GVB and FCI should agree for 2e/2o: {mc.e_tot} vs {mc_fci.e_tot}"


def test_rdm1_trace():
    """Check that the 1-RDM trace equals the number of electrons."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()

    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver = GVBSolver(npair=1)
    mc.verbose = 0
    mc.kernel()

    civec = mc.ci
    rdm1 = mc.fcisolver.make_rdm1(civec, 2, 2)

    assert abs(np.trace(rdm1) - 2.0) < 1e-10, \
        f"1-RDM trace should be 2, got {np.trace(rdm1)}"


def test_rdm12_consistency():
    """Check that rdm1 from make_rdm12 matches make_rdm1."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()

    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver = GVBSolver(npair=1)
    mc.verbose = 0
    mc.kernel()

    civec = mc.ci
    rdm1_direct = mc.fcisolver.make_rdm1(civec, 2, 2)
    rdm1_from_12, rdm2 = mc.fcisolver.make_rdm12(civec, 2, 2)

    assert np.allclose(rdm1_direct, rdm1_from_12, atol=1e-10), \
        "1-RDM from make_rdm1 and make_rdm12 should match"


def test_rdm2_symmetry():
    """Check 2-RDM symmetries."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()

    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver = GVBSolver(npair=1)
    mc.verbose = 0
    mc.kernel()

    civec = mc.ci
    rdm1, rdm2 = mc.fcisolver.make_rdm12(civec, 2, 2)

    # rdm2[p,q,r,s] = rdm2[r,s,p,q] (particle exchange symmetry)
    assert np.allclose(rdm2, rdm2.transpose(2, 3, 0, 1), atol=1e-10), \
        "2-RDM should have particle exchange symmetry"

    # Trace of rdm2 over one pair of indices gives (N-1)*rdm1
    # sum_r rdm2[p,q,r,r] = (N-1) * rdm1[p,q]
    N = 2
    rdm2_trace = np.einsum('pqrr->pq', rdm2)
    assert np.allclose(rdm2_trace, (N - 1) * rdm1, atol=1e-10), \
        "Partial trace of 2-RDM should give (N-1)*rdm1"


def test_h2_631g_casscf():
    """H2/6-31G: GVB(1) via CASSCF with larger basis."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    assert mf.converged

    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fcisolver = GVBSolver(npair=1)
    mc.verbose = 0
    mc.kernel()

    # Compare with FCI
    mc_fci = mcscf.CASSCF(mf, 2, 2)
    mc_fci.verbose = 0
    mc_fci.kernel()

    print(f"GVB-CASSCF/6-31G energy: {mc.e_tot:.8f}")
    print(f"FCI-CASSCF/6-31G energy: {mc_fci.e_tot:.8f}")

    # For 2e/2o, GVB(1) = FCI
    assert abs(mc.e_tot - mc_fci.e_tot) < 1e-5, \
        f"GVB and FCI should agree: {mc.e_tot} vs {mc_fci.e_tot}"


def test_energy_with_rdm():
    """Verify that energy computed from RDMs matches kernel energy."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()

    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver = GVBSolver(npair=1)
    mc.verbose = 0
    mc.kernel()

    # Get the CAS integrals
    h1, ecore = mc.get_h1eff()
    eri = mc.get_h2eff()
    from pyscf import ao2mo
    eri = ao2mo.restore(1, eri, 2)

    # Compute energy from RDMs
    civec = mc.ci
    rdm1, rdm2 = mc.fcisolver.make_rdm12(civec, 2, 2)

    e_from_rdm = ecore + np.einsum('pq,pq->', h1, rdm1) + \
                 0.5 * np.einsum('pqrs,pqrs->', eri, rdm2)

    print(f"Energy from kernel: {mc.e_tot:.10f}")
    print(f"Energy from RDMs:   {e_from_rdm:.10f}")

    assert abs(mc.e_tot - e_from_rdm) < 1e-6, \
        f"Energy from RDMs should match kernel: {mc.e_tot} vs {e_from_rdm}"


if __name__ == '__main__':
    test_h2_sto3g_casci()
    test_h2_sto3g_casscf()
    test_rdm1_trace()
    test_rdm12_consistency()
    test_rdm2_symmetry()
    test_h2_631g_casscf()
    test_energy_with_rdm()
    print("\nAll tests passed!")
