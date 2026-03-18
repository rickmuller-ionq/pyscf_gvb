# CLAUDE.md — Project context for hot start

## What this project is

A PySCF plugin that implements a GVB (Generalized Valence Bond) solver as a
drop-in `fcisolver` for PySCF's CASSCF/CASCI framework. Ported from PyQuante2's
`mcscf.py` by Rick Muller.

## Project layout

```
pyscf_gvb/
  __init__.py          # exports GVBSolver
  gvb_solver.py        # main implementation (~595 lines)
    - GVBCIVector       # opaque CI state container (pair_coeffs + orb_rotation)
    - GVBSolver         # fcisolver interface: kernel, make_rdm1, make_rdm12, spin_square
tests/
  test_gvb_solver.py   # 7 tests: H2 energy, RDM consistency, CASCI comparison, etc.
examples/
  01_h2_gvb.py         # H2/STO-3G example showing GVBSolver with CASSCF
setup.py               # minimal setuptools config
```

## Key design decisions

- **PySCF fcisolver interface**: `GVBSolver` implements `kernel(h1e, eri, norb, nelec, ci0, ecore)`,
  `make_rdm1(fcivec, norb, nelec)`, `make_rdm12(fcivec, norb, nelec)`, and
  `spin_square(fcivec, norb, nelec)`. This is the standard interface CASSCF expects.
- **GVBCIVector**: Opaque object returned by `kernel` and passed back to RDM methods.
  Stores `pair_coeffs` (shape `(npair, 2)`) and `orb_rotation` (unitary matrix).
- **2-RDM convention**: Chemists' notation `rdm2[p,q,r,s] = <p^+ r^+ s q>`, matching PySCF's expectation.
- **Orbital optimization**: Each GVB pair mixes two orbitals via 2x2 rotations.
  The solver iterates orbital rotations and CI coefficient updates to self-consistency.
- **Active space assumption**: `norb = 2 * npair`, `nelec = 2 * npair` (all singlet-paired electrons).

## How to run

```bash
# Install in dev mode (needs PySCF)
pip install -e .

# Run tests
python -m pytest tests/ -v

# Run example
python examples/01_h2_gvb.py
```

## Current status (as of initial port)

- All 7 tests pass (H2 energy, RDM trace/symmetry/idempotency, CASCI comparison, multi-pair, spin_square)
- Solver works for simple systems (H2, LiH)
- No Newton-CASSCF support yet (no `absorb_h1e`, `contract_2e`, `make_hdiag`, `transform_ci_for_orbital_rotation`)

## Potential next steps

- Add Newton-CASSCF compatible methods (`absorb_h1e`, `contract_2e`, etc.) for `mc2step` support
- Test on larger molecules and more complex active spaces
- Add open-shell / high-spin support (currently singlet-only)
- Performance optimization for larger pair counts
- Add geometry optimization / gradient support
- Integration tests against published GVB results from the Bobrowicz & Goddard reference

## Reference

Bobrowicz & Goddard, "The Self-Consistent Equations for Generalized Valence Bond
and Open-Shell Hartree-Fock Wave Functions", in Methods of Electronic Structure
Theory, Schaeffer ed., Plenum 1977.
http://www.wag.caltech.edu/publications/sup/pdf/108.pdf
