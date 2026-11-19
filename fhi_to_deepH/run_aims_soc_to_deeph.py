"""Convert one plain FHI-aims SOC calculation using the bundled legacy parser.

Usage: python run_aims_soc_to_deeph.py AIMS_DIRECTORY OUTPUT_DIRECTORY
Requires NumPy, h5py, ASE and DeepH-dock (deepx_dock). The historical parser
applies its legacy periodic same-orbital sign correction once. Use this route
for compatible legacy raw exports, including the source revision in this kit.
Input files are staged because the existing parser may repair basis indices.
Existing generated output files are overwritten; unrelated files are kept.
"""

import argparse
import math
from pathlib import Path
import shutil
import tempfile


def parse_soc_hamiltonian(aims_directory, output_directory, tolerance=1e-3, legacy_soc_sign="auto"):
    """Write spinor/scalar H, overlap and metadata without modifying raw inputs."""
    from aims_soc_to_deeph import FHIAimsReader, _find_soc_matrix_file

    source = Path(aims_directory).resolve(strict=True)
    destination = Path(output_directory).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    names = ["control.in", "geometry.in", "aims.out", "basis-indices.out",
             "rs_indices.out", "rs_hamiltonian.out", "rs_overlap.out"]
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    soc_file, _ = _find_soc_matrix_file(source)
    names.append(soc_file.name)

    with tempfile.TemporaryDirectory(prefix="aims_soc_conversion_") as scratch:
        stage = Path(scratch)
        for name in names:
            shutil.copy2(source / name, stage / name)
        reader = FHIAimsReader(stage, destination, export_H=True, export_H0=True,
                              tolerance=tolerance, legacy_soc_sign=legacy_soc_sign)
        result = reader.analysis_data()
        if result != 0:
            raise RuntimeError(f"FHI-aims parser returned status {result}")
        if not reader.spin_orbit or reader.spinful or reader.aims_data_type != "plain":
            raise ValueError("This runner requires a nonmagnetic, plain-output SOC calculation")
        # Create the destination after parsing succeeds. dump_data() overwrites
        # generated files within it and leaves unrelated files alone.
        destination.mkdir(parents=True, exist_ok=True)
        reader.dump_data()
        return len(reader.atom_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aims_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Existing parser's consistency tolerance in eV (default: 1e-3)")
    parser.add_argument(
        "--legacy-soc-sign",
        choices=("auto", "true", "false"),
        default="auto",
        help="Handle legacy (260530) same-sign R!=0 diagonal SOC bug: auto (default), true, or false",
    )
    args = parser.parse_args()
    blocks = parse_soc_hamiltonian(
        args.aims_directory,
        args.output_directory,
        args.tolerance,
        legacy_soc_sign=args.legacy_soc_sign,
    )
    print(f"Converted {blocks} atom-pair blocks to {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
