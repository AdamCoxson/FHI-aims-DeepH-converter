from __future__ import annotations

from pathlib import Path
import shutil
import struct
import sys
import unittest

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "fhi_to_deepH"))

import aims_soc_to_deeph as soc_mod  # noqa: E402
from run_aims_soc_to_deeph import parse_soc_hamiltonian  # noqa: E402


SIMPLE_MOS2_DIR = (
    Path("/home/adam/work/FHI-aims/FHI-aims_rs_soc_hamiltonian_modifications/simple_mos2_calc")
)
LEGACY_BINARY_DIR = Path(
    "/home/adam/work/ML-hamiltonians-workflow/sparsify_script_dev/sparsify_testing/"
    "example_data/fhi_DFT_data/MoS2_1x1_000_exx+0.0032_eyy+0.0078/soc_tight"
)


def _write_fortran_soc_bin(
    path: Path,
    data: np.ndarray,
    *,
    marker_bytes: int = 4,
    int_bytes: int = 4,
) -> None:
    ld, ncols = data.shape
    m_fmt = "<Q" if marker_bytes == 8 else "<I"
    i_fmt = "<q" if int_bytes == 8 else "<i"
    payload = np.asarray(data, dtype="<f8").tobytes(order="F")
    with open(path, "wb") as f:
        f.write(struct.pack(m_fmt, int_bytes))
        f.write(struct.pack(i_fmt, ld))
        f.write(struct.pack(m_fmt, int_bytes))

        f.write(struct.pack(m_fmt, int_bytes))
        f.write(struct.pack(i_fmt, ncols))
        f.write(struct.pack(m_fmt, int_bytes))

        f.write(struct.pack(m_fmt, len(payload)))
        f.write(payload)
        f.write(struct.pack(m_fmt, len(payload)))


def _create_synthetic_aims_dir(
    root: Path,
    *,
    soc_filename: str,
    binary: bool,
    same_sign_diag: bool,
    mixed_diag: bool = False,
    marker_bytes: int = 4,
    int_bytes: int = 4,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "geometry.in").write_text(
        "lattice_vector 3.0 0.0 0.0\n"
        "lattice_vector 0.0 3.0 0.0\n"
        "lattice_vector 0.0 0.0 10.0\n"
        "atom 0.0 0.0 0.0 H\n"
    )
    (root / "control.in").write_text(
        "xc pbe\n"
        "include_spin_orbit\n"
        "output_rs_matrices plain\n"
    )
    (root / "aims.out").write_text(
        "| Chemical potential (Fermi level):    -4.50000000 eV\n"
        "Have a nice day.\n"
    )
    (root / "basis-indices.out").write_text(
        "\n"
        " fn.   type   at.   n   l   m\n"
        "   1  atomic    1    1   0   0\n"
        "   2  atomic    1    2   0   0\n"
    )
    # 3 real cells (+ sentinel) = 4 lines; n_basis = 2; n_ham_size = 6 (+ sentinel) = 7
    # Cells: (0,0,0), (1,0,0), (-1,0,0), sentinel
    # Note _read_mx_indices negates cell_indices, so file (1,0,0) -> -1, (-1,0,0) -> +1.
    (root / "rs_indices.out").write_text(
        "n_hamiltonian_matrix_size: 7\n"
        "n_cells_in_hamiltonian: 4\n"
        "n_basis: 2\n"
        "cell_index\n"
        "   0   0   0\n"
        "   1   0   0\n"
        "  -1   0   0\n"
        " 99999999 99999999 99999999\n"
        "index_hamiltonian(1,:,:)\n"
        "   1   2\n"
        "   3   4\n"
        "   5   6\n"
        "   0   0\n"
        "index_hamiltonian(2,:,:)\n"
        "   1   2\n"
        "   3   4\n"
        "   5   6\n"
        "  -1  -1\n"
        "column_index_hamiltonian\n"
        "   1\n"
        "   2\n"
        "   1\n"
        "   2\n"
        "   1\n"
        "   2\n"
        "   0\n"
    )
    (root / "rs_overlap.out").write_text("1.0\n1.0\n0.1\n0.1\n0.1\n0.1\n0.0\n")
    (root / "rs_hamiltonian.out").write_text("-0.5\n-0.2\n-0.05\n-0.02\n-0.05\n-0.02\n0.0\n")

    # SOC components (Hartree):
    # row 0,1 = R=(0,0,0) ib=0,1; row 2,3 = nR=(-1,0,0) ib=0,1; row 4,5 = nR=(+1,0,0) ib=0,1; row 6 = sentinel
    val_x, val_y, val_z = 0.02, -0.03, 0.04
    sign = 1.0 if same_sign_diag else -1.0
    if mixed_diag:
        # ib=0 is same-sign, ib=1 is opposite-sign
        neg_row0 = [val_x, val_y, val_z]
        neg_row1 = [-val_x, -val_y, -val_z]
    else:
        # In legacy same-sign bug, both nR=(-1,0,0) and nR=(+1,0,0) store +val.
        # In fixed Nov26, nR=(-1,0,0) stores -val and nR=(+1,0,0) stores +val.
        neg_row0 = [sign * val_x, sign * val_y, sign * val_z]
        neg_row1 = [sign * val_x, sign * val_y, sign * val_z]

    soc_arr = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            neg_row0,
            [0.5 * x for x in neg_row1],
            [val_x, val_y, val_z],
            [0.5 * val_x, 0.5 * val_y, 0.5 * val_z],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    if binary:
        _write_fortran_soc_bin(
            root / soc_filename,
            soc_arr,
            marker_bytes=marker_bytes,
            int_bytes=int_bytes,
        )
    else:
        with open(root / soc_filename, "w") as f:
            f.write(f"{soc_arr.shape[0]} {soc_arr.shape[1]}\n")
            for row in soc_arr:
                f.write(f"{row[0]:26.16E}{row[1]:26.16E}{row[2]:26.16E}\n")


def _read_deeph_blocks(h5_path: Path) -> dict[tuple[int, int, int, int, int], np.ndarray]:
    blocks = {}
    with h5py.File(h5_path, "r") as f:
        pairs = f["atom_pairs"][:]
        bounds = f["chunk_boundaries"][:]
        shapes = f["chunk_shapes"][:]
        entries = f["entries"][:]
    for idx, pair in enumerate(pairs):
        key = tuple(int(x) for x in pair)
        r, c = int(shapes[idx, 0]), int(shapes[idx, 1])
        blocks[key] = entries[bounds[idx] : bounds[idx + 1]].reshape((r, c))
    return blocks


def _max_hermiticity_error(blocks: dict[tuple[int, int, int, int, int], np.ndarray]) -> float:
    max_err = 0.0
    for (rx, ry, rz, ia, ja), mat in blocks.items():
        partner = blocks[(-rx, -ry, -rz, ja, ia)]
        err = float(np.max(np.abs(mat - partner.conj().T)))
        if err > max_err:
            max_err = err
    return max_err


class AimsSocToDeephTests(unittest.TestCase):
    def test_binary_reader_autodetects_markers_and_header_ints(self):
        import tempfile

        rng = np.random.default_rng(42)
        expected = rng.normal(size=(7, 3))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for marker_bytes in (4, 8):
                for int_bytes in (4, 8):
                    with self.subTest(marker_bytes=marker_bytes, int_bytes=int_bytes):
                        bin_path = tmp_path / f"rs_soc_m{marker_bytes}_i{int_bytes}.bin"
                        _write_fortran_soc_bin(
                            bin_path,
                            expected,
                            marker_bytes=marker_bytes,
                            int_bytes=int_bytes,
                        )
                        parsed = soc_mod._read_soc_binary_raw(bin_path, n_ham_size=7)
                        np.testing.assert_allclose(parsed, expected, atol=0.0, rtol=0.0)

    def test_binary_preferred_over_plain_and_matches_real_legacy_data(self):
        if not LEGACY_BINARY_DIR.is_dir():
            self.skipTest(f"Legacy binary fixture not present: {LEGACY_BINARY_DIR}")
        px_b, py_b, pz_b = soc_mod._read_soc_matrices(LEGACY_BINARY_DIR, 148100)
        plain_ha = soc_mod._read_soc_plain_raw(
            LEGACY_BINARY_DIR / "realspace_soc_matrix.out", 148100
        )
        np.testing.assert_allclose(px_b, plain_ha[:, 0] * soc_mod.HARTREE_TO_EV, atol=1e-12)
        np.testing.assert_allclose(py_b, plain_ha[:, 1] * soc_mod.HARTREE_TO_EV, atol=1e-12)
        np.testing.assert_allclose(pz_b, plain_ha[:, 2] * soc_mod.HARTREE_TO_EV, atol=1e-12)

    def test_auto_detects_legacy_vs_nov26_and_rejects_mixed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1) Legacy same-sign (realspace_soc_matrix_unfmttd.out)
            leg_dir = tmp_path / "legacy_bin"
            leg_out = tmp_path / "legacy_out"
            _create_synthetic_aims_dir(
                leg_dir,
                soc_filename="realspace_soc_matrix_unfmttd.out",
                binary=True,
                same_sign_diag=True,
            )
            reader_leg = soc_mod.FHIAimsReader(
                leg_dir, leg_out, export_H=True, export_H0=True, legacy_soc_sign="auto"
            )
            self.assertEqual(reader_leg.analysis_data(), 0)
            self.assertTrue(reader_leg.detected_legacy_soc_sign)
            reader_leg.dump_data()

            # 2) Nov26 opposite-sign (rs_soc_matrix.bin, 8-byte markers, Int64 header)
            nov_dir = tmp_path / "nov26_bin"
            nov_out = tmp_path / "nov26_out"
            _create_synthetic_aims_dir(
                nov_dir,
                soc_filename="rs_soc_matrix.bin",
                binary=True,
                same_sign_diag=False,
                marker_bytes=8,
                int_bytes=8,
            )
            reader_nov = soc_mod.FHIAimsReader(
                nov_dir, nov_out, export_H=True, export_H0=True, legacy_soc_sign="auto"
            )
            self.assertEqual(reader_nov.analysis_data(), 0)
            self.assertFalse(reader_nov.detected_legacy_soc_sign)
            reader_nov.dump_data()

            leg_blocks = _read_deeph_blocks(leg_out / "soc-hamiltonian.h5")
            nov_blocks = _read_deeph_blocks(nov_out / "soc-hamiltonian.h5")
            self.assertLess(_max_hermiticity_error(leg_blocks), 1e-14)
            self.assertLess(_max_hermiticity_error(nov_blocks), 1e-14)
            self.assertEqual(set(leg_blocks.keys()), set(nov_blocks.keys()))
            for k in leg_blocks:
                np.testing.assert_allclose(leg_blocks[k], nov_blocks[k], atol=1e-14)

            # 3) Mixed signs -> ValueError in auto mode
            mix_dir = tmp_path / "mixed"
            mix_out = tmp_path / "mixed_out"
            _create_synthetic_aims_dir(
                mix_dir,
                soc_filename="rs_soc_matrix.out",
                binary=False,
                same_sign_diag=False,
                mixed_diag=True,
            )
            reader_mix = soc_mod.FHIAimsReader(
                mix_dir, mix_out, export_H=True, export_H0=True, legacy_soc_sign="auto"
            )
            with self.assertRaises(ValueError):
                reader_mix.analysis_data()

    def test_simple_mos2_calc_parity_auto_vs_true(self):
        if not SIMPLE_MOS2_DIR.is_dir():
            self.skipTest(f"simple_mos2_calc fixture not present: {SIMPLE_MOS2_DIR}")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_auto = tmp_path / "out_auto"
            out_true = tmp_path / "out_true"

            n_auto = parse_soc_hamiltonian(SIMPLE_MOS2_DIR, out_auto, legacy_soc_sign="auto")
            n_true = parse_soc_hamiltonian(SIMPLE_MOS2_DIR, out_true, legacy_soc_sign="true")
            self.assertEqual(n_auto, n_true)

            b_auto = _read_deeph_blocks(out_auto / "soc-hamiltonian.h5")
            b_true = _read_deeph_blocks(out_true / "soc-hamiltonian.h5")
            self.assertLess(_max_hermiticity_error(b_auto), 1e-10)
            for k in b_auto:
                np.testing.assert_allclose(b_auto[k], b_true[k], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
