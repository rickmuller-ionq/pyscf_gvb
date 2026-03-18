"""
pyscf_gvb - A GVB (Generalized Valence Bond) solver for PySCF.

Ported from PyQuante2's GVB solver by Rick Muller.
Provides a GVBSolver class that can be used as mc.fcisolver in PySCF's
CASSCF/CASCI framework.

Usage:
    from pyscf import gto, scf, mcscf
    from pyscf_gvb import GVBSolver

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fcisolver = GVBSolver(mol)
    mc.kernel()
"""

from pyscf_gvb.gvb_solver import GVBSolver

__all__ = ['GVBSolver']
