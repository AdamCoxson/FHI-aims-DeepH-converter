"""Run with: python -m unittest discover -s python_build_spinor_hamiltonian -v (NumPy only)."""

import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

UTILITY = Path(__file__).resolve().parent / "build_spinor_hamiltonian.py"
spec = importlib.util.spec_from_file_location("spinor_utility", UTILITY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build_spinor_hamiltonian = module.build_spinor_hamiltonian


class SpinorTests(unittest.TestCase):
    def test_hand_calculated_signs(self):
        result = build_spinor_hamiltonian([[2]], [[3]], [[5]], [[7]])
        assert_array_equal(result, [[2-7j, -5-3j], [5-3j, 2+7j]])
        self.assertEqual(result.dtype, np.complex128)

    def test_rectangular_order_and_zero_soc(self):
        h = np.arange(6).reshape(2, 3)
        zeros = np.zeros_like(h)
        result = build_spinor_hamiltonian(h, zeros, zeros, zeros)
        assert_array_equal(result, np.block([[h, zeros], [zeros, h]]))
        px, py, pz = h + 1, h + 10, h + 100
        result = build_spinor_hamiltonian(h, px, py, pz)
        self.assertEqual(result.shape, (4, 6))
        assert_array_equal(result[:2, 3:], -py - 1j*px)
        assert_array_equal(result[2:, :3], py - 1j*px)
        assert_array_equal(result[:2, :3], h - 1j*pz)
        assert_array_equal(result[2:, 3:], h + 1j*pz)

    def test_onsite_hermiticity(self):
        h = np.array([[2, 3], [3, 4]])
        p = np.array([[0, 5], [-5, 0]])
        result = build_spinor_hamiltonian(h, p, p*2, p*3)
        assert_array_equal(result, result.conj().T)

    def test_independent_translation_partners(self):
        h = np.arange(6).reshape(2, 3)
        p = h + 2
        forward = build_spinor_hamiltonian(h, p, 2*p, 3*p)
        backward = build_spinor_hamiltonian(h.T, -p.T, -2*p.T, -3*p.T)
        assert_array_equal(forward, backward.conj().T)
        # A nonzero-R block itself need not be Hermitian.
        self.assertNotEqual(forward.shape[0], forward.shape[1])

    def test_legacy_same_orbital_pair(self):
        h, px, py, pz = ([[v]] for v in [2, 3, 5, 7])
        positive = build_spinor_hamiltonian(h, px, py, pz)
        uncorrected_negative = build_spinor_hamiltonian(h, px, py, pz)
        self.assertGreater(np.max(np.abs(positive - uncorrected_negative.conj().T)), 0)
        corrected_negative = build_spinor_hamiltonian(
            h, px, py, pz, legacy_antisymmetry=True, legacy_flip_mask=[[True]])
        assert_array_equal(positive, corrected_negative.conj().T)

    def test_only_masked_soc_changes_and_inputs_preserved(self):
        arrays = [np.arange(6, dtype=float).reshape(2, 3) + v for v in (0, 1, 2, 3)]
        mask = np.array([[True, False, False], [False, False, True]])
        snapshots = [a.copy() for a in arrays] + [mask.copy()]
        for a in [*arrays, mask]:
            a.setflags(write=False)
        corrected = build_spinor_hamiltonian(
            *arrays, legacy_antisymmetry=True, legacy_flip_mask=mask)
        expected = build_spinor_hamiltonian(arrays[0], *(np.where(mask, -a, a) for a in arrays[1:]))
        assert_array_equal(corrected, expected)
        for a, saved in zip([*arrays, mask], snapshots):
            assert_array_equal(a, saved)
        normal = build_spinor_hamiltonian(*arrays)
        self.assertFalse(any(np.shares_memory(normal, a) for a in arrays))

    def test_invalid_values_and_shapes(self):
        good = np.ones((2, 3))
        for bad in (np.ones((3, 2)), np.ones(6), np.ones((0, 3)), np.ones((2, 0)),
                    np.ones((2, 3, 1)), [[np.nan]*3]*2, [[np.inf]*3]*2):
            for position in range(4):
                with self.subTest(shape=np.shape(bad), position=position):
                    args = [good]*4
                    args[position] = bad
                    with self.assertRaises(ValueError):
                        build_spinor_hamiltonian(*args)

    def test_reject_complex_and_nonnumeric(self):
        for bad in ([[1+0j]], [[True]], [["1"]], [[object()]]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                build_spinor_hamiltonian(bad, [[0]], [[0]], [[0]])

    def test_mask_contract(self):
        args = [np.ones((2, 3))]*4
        with self.assertRaises(ValueError):
            build_spinor_hamiltonian(*args, legacy_antisymmetry=True)
        for mask in (np.ones((2, 3)), [[1, 0, 0], [0, 0, 1]]):
            with self.assertRaises(TypeError):
                build_spinor_hamiltonian(*args, legacy_antisymmetry=True, legacy_flip_mask=mask)
        for mask in (True, [True]*6, [[True]], np.ones((3, 2), dtype=bool)):
            with self.assertRaises(ValueError):
                build_spinor_hamiltonian(*args, legacy_antisymmetry=True, legacy_flip_mask=mask)
        with self.assertRaises(ValueError):
            build_spinor_hamiltonian(*args, legacy_flip_mask=np.zeros((2, 3), dtype=bool))
        with self.assertRaises(TypeError):
            build_spinor_hamiltonian(*args, legacy_antisymmetry="False")

    def test_standalone_examples(self):
        tree = ast.parse(UTILITY.read_text())
        self.assertEqual([n.names[0].name for n in tree.body if isinstance(n, ast.Import)], ["numpy"])
        self.assertFalse(any(isinstance(n, ast.ImportFrom) for n in tree.body))
        scope = dict(build_spinor_hamiltonian=build_spinor_hamiltonian,
                     h_scalar=[[2]], p_x=[[3]], p_y=[[5]], p_z=[[7]],
                     mask_from_legacy_indices=[[True]])
        exec("h_spinor = build_spinor_hamiltonian(h_scalar, p_x, p_y, p_z)", scope)
        exec("h_spinor_legacy = build_spinor_hamiltonian(h_scalar, p_x, p_y, p_z, "
             "legacy_antisymmetry=True, legacy_flip_mask=mask_from_legacy_indices)", scope)
        assert_array_equal(scope["h_spinor"], scope["h_spinor_legacy"].conj().T)


if __name__ == "__main__":
    unittest.main()
