"""Convert one plain FHI-aims SOC calculation using the bundled legacy parser.

Usage: python run_aims_soc_to_deeph.py AIMS_DIRECTORY NEW_OUTPUT_DIRECTORY
Requires NumPy, h5py, ASE and DeepH-dock (deepx_dock). The historical parser
applies its legacy periodic same-orbital sign correction once. Use this route
for compatible legacy raw exports, including the source revision in this kit.
Input files are staged because the existing parser may repair basis indices.
"""

import argparse
import math
from pathlib import Path
import shutil
import tempfile


def parse_soc_hamiltonian(aims_directory, output_directory, tolerance=1e-3):
    """Write spinor/scalar H, overlap and metadata without modifying raw inputs."""
    from aims_soc_to_deeph import FHIAimsReader

    source = Path(aims_directory).resolve(strict=True)
    destination = Path(output_directory).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    names = ("control.in", "geometry.in", "aims.out", "basis-indices.out",
             "rs_indices.out", "rs_hamiltonian.out", "rs_overlap.out",
             "realspace_soc_matrix.out")
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    with tempfile.TemporaryDirectory(prefix="aims_soc_conversion_") as scratch:
        stage = Path(scratch)
        for name in names:
            shutil.copy2(source / name, stage / name)
        reader = FHIAimsReader(stage, destination, export_H=True, export_H0=True,
                              tolerance=tolerance)
        result = reader.analysis_data()
        if result != 0:
            raise RuntimeError(f"FHI-aims parser returned status {result}")
        if not reader.spin_orbit or reader.spinful or reader.aims_data_type != "plain":
            raise ValueError("This runner requires a nonmagnetic, plain-output SOC calculation")
        # Reserve the destination only after parsing succeeds; never delete an
        # existing directory. dump_data() accepts this newly created directory.
        destination.mkdir(parents=True, exist_ok=False)
        reader.dump_data()
        return len(reader.atom_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aims_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Existing parser's consistency tolerance in eV (default: 1e-3)")
    args = parser.parse_args()
    blocks = parse_soc_hamiltonian(args.aims_directory, args.output_directory, args.tolerance)
    print(f"Converted {blocks} atom-pair blocks to {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
