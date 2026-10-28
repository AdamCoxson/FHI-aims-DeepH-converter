# FHI-aims to DeepH converter
Python scripts for use with FHI-aims output to convert real space Hamiltonians into DeepH compatible format for ML-Hamiltonian training

Convert real-space matrices from a periodic, nonmagnetic FHI-aims calculation
with spin–orbit coupling (SOC) into the atom-pair HDF5 format used by DeepH-dock.
The conversion produces a complex spinor Hamiltonian, a scalar baseline,
overlap matrices, and structure/basis metadata for downstream ML Hamiltonian
workflows.

This repository contains two tools:

| Tool | Purpose |
| --- | --- |
| [`fhi_to_deepH/run_aims_soc_to_deeph.py`](fhi_to_deepH/run_aims_soc_to_deeph.py) | Convert one calculation directory using the bundled parser. Start here for exported FHI-aims files. |
| [`build_spinor_hamiltonian.py`](build_spinor_hamiltonian.py) | Assemble a spinor matrix from four already unpacked NumPy arrays. |

The accompanying [`test_build_spinor_hamiltonian.py`](test_build_spinor_hamiltonian.py)
checks the full spinor assembly; see [Run the spinor tests](#run-the-spinor-tests).

## 1. Install

Clone the repository and create an environment using Python 3.12–3.14:

```bash
git clone https://github.com/AdamCoxson/FHI-aims-DeepH-converter.git
cd FHI-aims-DeepH-converter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy h5py ase "deepx-dock==0.9.11"
```

DeepH-dock is installed as `deepx-dock` and imported as `deepx_dock`. Its
[installation guide](https://docs.deeph-pack.com/deeph-dock/en/latest/installation_and_setup.html)
covers other installation methods. Version `0.9.11` is the version checked
with this converter; the scripts import internal DeepH-dock modules, so newer
versions may need compatibility checks.

Check that the runner and its imports work:

```bash
python fhi_to_deepH/run_aims_soc_to_deeph.py --help
python -c "import sys; sys.path.insert(0, 'fhi_to_deepH'); from aims_soc_to_deeph import FHIAimsReader; print('Imports OK')"
```

Run the scripts directly from the clone; there is no package installation step
for this repository. The array-only helper requires just NumPy.

## 2. Prepare the FHI-aims calculation

You need an FHI-aims build that exports the raw real-space SOC components as
`realspace_soc_matrix.out`, alongside the scalar Hamiltonian and overlap.
The converter was validated with legacy raw exports from a modified FHI-aims
revision, `f68eead427b68dbf1828261d9872383a523f9ac4`. The FHI-aims source
modifications and executable are **not included in this repository**.

For a compatible build, include these settings in `control.in`, in addition
to your normal calculation settings and species definitions:

```text
include_spin_orbit
output_rs_matrices plain
output rs_soc_matrix plain
```

The legacy SOC exporter requires global matrix storage; its compatible setup
uses `use_local_index .false.` and `load_balancing .false.`. Other restrictions
on real-space matrix export depend on your FHI-aims build. If
`output rs_soc_matrix plain` is unrecognized, obtain a build with that exporter
before using this conversion route.

Run the calculation and retain its standard output as `aims.out`. All eight
files below must come from the **same calculation**, with matching geometry,
basis and sparse matrix indexing:

```text
my_calculation/
├── control.in
├── geometry.in
├── aims.out
├── basis-indices.out
├── rs_indices.out
├── rs_hamiltonian.out
├── rs_overlap.out
└── realspace_soc_matrix.out
```

`realspace_soc_matrix.out` contains three columns, the x, y and z SOC
components in Hartree, in the packed order described by `rs_indices.out`.
The parser accepts an optional first-line header containing the packed length
and `3`. A binary SOC export is not an input to this runner.

The command-line runner supports **periodic, nonmagnetic SOC calculations
with plain-text matrix output**. Do not use `spin collinear` for this route.
Molecules without periodic lattice vectors, magnetic SOC calculations, and
HDF5 input matrices are outside the runner's supported scope.

### Legacy SOC convention

The bundled parser always applies a historical sign correction to explicitly
stored SOC entries for the same global orbital at nonzero lattice translation.
It flips the entries whose first nonzero translation component is negative
in the converter's translation convention. The converter uses the negative
of the FHI-aims cell indices.

Use this parser with compatible **uncorrected legacy raw exports**. Applying
it to exports whose signs have already been corrected upstream or in another
post-processing step can introduce an error. The runner has no switch to
disable this correction.

## 3. Convert one calculation

From the repository root, run:

```bash
python fhi_to_deepH/run_aims_soc_to_deeph.py \
    /path/to/my_calculation \
    /path/to/converted_calculation
```

Replace both paths with your own. The output directory must **not already
exist**; the runner creates it and any missing parents. It copies the eight
input files to a temporary directory before parsing, so automatic basis-index
repairs leave your original calculation files unchanged.

On success, it prints `Converted ... atom-pair blocks to ...`.

The optional consistency tolerance defaults to `1e-3` eV:

```bash
python fhi_to_deepH/run_aims_soc_to_deeph.py \
    /path/to/my_calculation \
    /path/to/another_new_output \
    --tolerance 1e-3
```

This controls consistency checks between stored symmetry partners during
unpacking. It must be finite and positive; it is not an SCF convergence
threshold or a matrix truncation cutoff.

Run the command once per calculation, choosing a separate output directory
for each structure.

## 4. Understand the outputs

| File | Contents |
| --- | --- |
| `soc-hamiltonian.h5` | Full complex SOC spinor Hamiltonian, in eV. This is the SOC training target. |
| `hamiltonian.h5` | Scalar `rs_hamiltonian.out` baseline copied onto both spin diagonals, with zero spin-off-diagonal blocks, in eV. |
| `overlap.h5` | Real, dimensionless spatial overlap, without spin doubling. |
| `info.json` | Atom and spatial-orbital counts, orbital angular momenta by element, spin flags, occupation and Fermi energy. |
| `POSCAR` | Structure in Cartesian coordinates, in Å, with atoms grouped alphabetically by chemical symbol. |

**Check the Hamiltonian filename in your downstream workflow.** A consumer
that reads `hamiltonian.h5` by default will load the scalar baseline. Configure
it to use `soc-hamiltonian.h5`, or prepare a separate copy of the output with
the full SOC file under the filename that consumer expects.

The baseline is taken from `rs_hamiltonian.out`; this SOC runner does not
require `rs_hamiltonian0.out`. Energies are converted using
`1 Hartree = 27.2113845 eV`. The Fermi energy is recorded as metadata, without
subtracting it from the Hamiltonian. If the expected chemical-potential line
is absent from `aims.out`, the parser records a Fermi energy of `0.0`.

Within each atom-pair Hamiltonian block, all spin-up orbitals precede all
spin-down orbitals. A pair with spatial dimensions `(n, m)` therefore has
Hamiltonian dimensions `(2*n, 2*m)` and overlap dimensions `(n, m)`. When
assembling a spinor overlap, use `diag(S, S)` with the same ordering.

Atoms are remapped to match the output POSCAR, periodic positions are mapped
to the central cell, and orbital phases are converted by the parser. Use the
output structure and indexing together.

### Read an HDF5 block

Each matrix file stores four datasets:

| Dataset | Meaning |
| --- | --- |
| `atom_pairs` | Rows `[R1, R2, R3, i, j]`: integer lattice translation and zero-based output atom indices. |
| `chunk_boundaries` | Start/end offsets into `entries`, with one more offset than there are atom pairs. |
| `chunk_shapes` | Row and column counts for each atom-pair block. |
| `entries` | All blocks flattened in NumPy row-major order and concatenated. |

For example, from the directory containing your converted output:

```python
import h5py

with h5py.File("converted_calculation/soc-hamiltonian.h5", "r") as f:
    k = 0  # Select an atom-pair block.
    pair = f["atom_pairs"][k]
    start, end = f["chunk_boundaries"][k:k + 2]
    shape = tuple(f["chunk_shapes"][k])
    block = f["entries"][start:end].reshape(shape)

print("[R1, R2, R3, i, j]:", pair)
print("Block shape:", block.shape)
```

## Array-only spinor assembly

If you already have unpacked scalar and SOC matrices, use the independent
NumPy helper from the repository root:

```python
import numpy as np
from build_spinor_hamiltonian import build_spinor_hamiltonian

h_scalar = np.array([[1.0, 0.2], [0.2, 2.0]])
p_x = np.zeros((2, 2))
p_y = np.zeros((2, 2))
p_z = np.array([[0.0, 0.01], [-0.01, 0.0]])

h_spinor = build_spinor_hamiltonian(h_scalar, p_x, p_y, p_z)
print(h_spinor.shape)  # (4, 4), complex128
```

The helper constructs:

```text
            [ h_scalar - i*p_z    -i*p_x - p_y     ]
H_spinor =  [                                     ]
            [ -i*p_x + p_y        h_scalar + i*p_z ]
```

Inputs must be finite, real, nonempty two-dimensional arrays of identical
shape, with matching units, orbital order and phases. Rectangular atom-pair
blocks are supported. The result uses the input energy units and leaves the
inputs unchanged. The helper does not parse export files, unpack sparse
vectors, convert units, or Fourier-sum lattice translations.

Its legacy correction defaults to **off**. Leave it off for components already
corrected by the converter. For uncorrected legacy components, the optional
`legacy_antisymmetry=True` mode requires a boolean `legacy_flip_mask` selecting
the affected explicitly stored entries from sparse metadata. See the
[function docstring](build_spinor_hamiltonian.py) for the exact mask convention;
a local block diagonal alone does not identify the affected global orbitals.

## Run the spinor tests

From the repository root, with your Python environment activated, run:

```bash
python -m unittest -v test_build_spinor_hamiltonian
```

The tests require only NumPy and Python's built-in `unittest` module. No
FHI-aims executable, calculation files, or DeepH-dock installation is needed.
A successful run reports `Ran 10 tests` followed by `OK`.

The suite checks:

- All four spin blocks against hand-calculated signs, including spin ordering,
  rectangular blocks, zero SOC, and the `complex128` output type.
- Onsite Hermiticity and the relation `H(R) = H(-R).conj().T` between matching
  translation partners.
- The explicit legacy sign-correction mask, including that only selected SOC
  entries change and input arrays remain unchanged.
- Rejection of invalid shapes, nonfinite or complex inputs, and invalid masks,
  plus the standalone helper examples and its NumPy-only dependency.

These are unit tests of `build_spinor_hamiltonian`. They do not validate file
parsing, the exported HDF5 files, or agreement with FHI-aims eigenvalues.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate the environment and install the dependencies with that environment's `python -m pip`. Check the `deepx_dock` spelling and version. |
| `Choose a new output directory` | Select a destination that does not exist yet. |
| Missing input file | Collect all eight required files. Rename the calculation's standard-output log to `aims.out` if needed. |
| Missing or unsupported `output_rs_matrices` | Use a compatible calculation with `output_rs_matrices plain` in `control.in`. |
| `Only periodic system is supported!` | Supply a periodic geometry with lattice vectors. |
| Atom, basis or matrix size mismatch | Check that all files came from the same calculation and are complete. |
| `Hermitian check failed` | Check input consistency and the legacy SOC convention before changing the tolerance. |

## License

[MIT](LICENSE).

