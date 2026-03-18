"""
Example: GVB(1) calculation for H2 using PySCF's CASSCF with GVBSolver.

This demonstrates how to use the GVB solver as a drop-in replacement
for PySCF's FCI solver in CASSCF/CASCI calculations.

For H2 with 1 GVB pair (2 active orbitals, 2 active electrons),
the GVB wavefunction is:
    |GVB> = c1 |sigma sigma_bar> + c2 |sigma* sigma*_bar>

This exactly recovers the FCI energy for 2 electrons in 2 orbitals.
"""

from pyscf import gto, scf, mcscf
from pyscf_gvb import GVBSolver

# Build molecule
mol = gto.M(
    atom='H 0 0 0; H 0 0 0.74',
    basis='6-31g',
    verbose=4,
)

# Run RHF
mf = scf.RHF(mol).run()
print(f"\nRHF energy: {mf.e_tot:.10f}")

# Run CASSCF with GVB solver (1 pair = 2 orbitals, 2 electrons)
mc = mcscf.CASSCF(mf, 2, 2)
mc.fcisolver = GVBSolver(npair=1)
mc.kernel()

print(f"\nGVB-CASSCF energy: {mc.e_tot:.10f}")
print(f"Correlation energy: {mc.e_tot - mf.e_tot:.10f}")

# Compare with standard FCI-CASSCF
mc_fci = mcscf.CASSCF(mf, 2, 2)
mc_fci.kernel()
print(f"FCI-CASSCF energy: {mc_fci.e_tot:.10f}")

# Show GVB pair coefficients
civec = mc.ci
print(f"\nGVB pair coefficients: {civec.pair_coeffs[0]}")
print(f"Natural orbital occupations: {2*civec.pair_coeffs[0]**2}")
