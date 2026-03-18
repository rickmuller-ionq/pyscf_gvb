# pyscf_gvb

A GVB (Generalized Valence Bond) solver for PySCF, ported from
[PyQuante2](https://github.com/rpmuller/pyquante2/blob/master/src/pyquante2/scf/mcscf.py).

Provides `GVBSolver`, a class that implements PySCF's `fcisolver` interface
and can be used as a drop-in replacement in `CASSCF` or `CASCI` calculations.

## Installation

```bash
pip install -e .
```

## Usage

```python
from pyscf import gto, scf, mcscf
from pyscf_gvb import GVBSolver

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g')
mf = scf.RHF(mol).run()

# CASSCF with 1 GVB pair (2 active orbitals, 2 active electrons)
mc = mcscf.CASSCF(mf, 2, 2)
mc.fcisolver = GVBSolver(npair=1)
mc.kernel()
```

## Reference

F.W. Bobrowicz and W.A. Goddard III,
"The Self-Consistent Equations for Generalized Valence Bond and
Open-Shell Hartree-Fock Wave Functions",
in *Methods of Electronic Structure Theory*, H.F. Schaeffer III, ed.
Plenum Publishing Corp., 1977.
