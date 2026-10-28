"""Assemble real-space spinor blocks from unpacked scalar and SOC matrices."""

import numpy as np

__all__ = ["build_spinor_hamiltonian"]


def build_spinor_hamiltonian(
    h_scalar, p_x, p_y, p_z, *, legacy_antisymmetry=False, legacy_flip_mask=None
):
    """Return the complex spinor Hamiltonian for one real-space matrix/block.

    Parameters
    ----------
    h_scalar, p_x, p_y, p_z : real array-like, each shape (n, m)
        Finite, nonempty, two-dimensional matrices of identical shape. They
        must already share energy units, orbital order and phase conventions.
        Complex inputs are rejected. The scalar baseline is shared by both
        spins (the nonmagnetic convention); separate collinear baselines are
        outside this interface.
    legacy_antisymmetry : bool, default False
        False is the ordinary path for already correct components. True flips
        all three SOC components at the entries selected by legacy_flip_mask,
        leaving h_scalar unchanged. Historical exports stored some periodic
        same-orbital partners symmetrically instead of antisymmetrically.
        Enabling this on upstream-corrected or already post-processed data can
        introduce a sign error. This is not a general Hermiticity repair.
    legacy_flip_mask : boolean array-like, shape (n, m), optional
        Required exactly when legacy_antisymmetry is True. The caller must
        select only uncorrected, explicitly stored legacy entries whose row
        and column identify the same global orbital (hence the same atom),
        with nonzero translation R and first nonzero component of R negative.
        The existing converter's _read_mx_indices returns the NEGATIVE of the
        FHI-aims cell indices: construct the mask in that converter convention.
        Local diagonal positions, especially in rectangular/inter-atom blocks,
        do not establish global orbital identity. Do not select partners already
        synthesized with the correct antisymmetric sign during unpacking, or
        entries already corrected by the parser. Onsite R=0, different-orbital
        and unselected entries stay untouched. Nonboolean, missing, incorrectly
        shaped and unused masks are errors; no mask is inferred from values.

    Returns
    -------
    numpy.ndarray, shape (2*n, 2*m), dtype complex128
        A new array in the input energy units. Rows and columns contain all
        spin-up orbitals followed by all spin-down orbitals, with blocks::

            [ h_scalar - 1j*p_z    -1j*p_x - p_y   ]
            [ -1j*p_x + p_y        h_scalar + 1j*p_z ]

        Square (N, N) inputs give a (2*N, 2*N) matrix; rectangular atom-pair
        blocks are also supported. For periodic models apply this per R/atom
        pair; it does not Fourier-sum translations. Hermiticity relates R to
        the conjugate transpose at -R, so individual blocks are not symmetrized.

    Notes
    -----
    All supplied arrays, including read-only arrays and the mask, are preserved.
    These are unpacked matrices, not raw one-dimensional packed export vectors.
    Export-file parsing and unpacking using sparse metadata remain the caller's
    responsibility. Components corrected by _trans_mxs_to_R_dict must use the
    default legacy_antisymmetry=False, avoiding a second correction.

    Examples
    --------
    Normal call, after unpacking the four matching component matrices::

        h_spinor = build_spinor_hamiltonian(h_scalar, p_x, p_y, p_z)

    Legacy call, after constructing a mask from the caller's sparse metadata::

        h_spinor_legacy = build_spinor_hamiltonian(
            h_scalar, p_x, p_y, p_z,
            legacy_antisymmetry=True,
            legacy_flip_mask=mask_from_legacy_indices,
        )
    """
    if not isinstance(legacy_antisymmetry, (bool, np.bool_)):
        raise TypeError("legacy_antisymmetry must be boolean")

    matrices = []
    for name, value in zip(("h_scalar", "p_x", "p_y", "p_z"), (h_scalar, p_x, p_y, p_z)):
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
        array = np.asarray(array, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        if matrices and array.shape != matrices[0].shape:
            raise ValueError("all four matrices must have identical shapes")
        matrices.append(array)

    h, px, py, pz = matrices
    if legacy_antisymmetry:
        if legacy_flip_mask is None:
            raise ValueError("legacy_antisymmetry requires legacy_flip_mask")
        mask = np.asarray(legacy_flip_mask)
        if mask.dtype.kind != "b":
            raise TypeError("legacy_flip_mask must contain boolean values")
        if mask.shape != h.shape:
            raise ValueError("legacy_flip_mask must have the same shape as the matrices")
        px, py, pz = (np.where(mask, -component, component) for component in (px, py, pz))
    elif legacy_flip_mask is not None:
        raise ValueError("legacy_flip_mask requires legacy_antisymmetry=True")

    n_rows, n_cols = h.shape
    spinor = np.empty((2 * n_rows, 2 * n_cols), dtype=np.complex128)
    spinor[:n_rows, :n_cols] = h - 1j * pz
    spinor[n_rows:, n_cols:] = h + 1j * pz
    spinor[:n_rows, n_cols:] = -1j * px - py
    spinor[n_rows:, :n_cols] = -1j * px + py
    return spinor
