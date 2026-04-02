from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ....aie_types import FloatFormat
from ....quant_utils import apply_rounding, dtype_for_precision, handle_overflow

TILING_OPTIONS: Dict[str, Dict[Tuple[Any, Any], List[Tuple[int, int, int]]]] = {
    'AIE': {
        (8, 8): [(2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 8, 8), (4, 16, 4), (4, 16, 8), (8, 8, 4)],
        (16, 8): [(4, 4, 4), (4, 4, 8), (4, 8, 4), (8, 4, 4)],
        (8, 16): [(4, 4, 8), (4, 4, 4), (8, 8, 1)],
        (16, 16): [(4, 4, 8), (2, 4, 8), (4, 2, 8), (4, 4, 4), (8, 8, 1)],
        ('float', 'float'): [(2, 4, 4)],
    },
    'AIE-ML': {
        (8, 8): [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8), (8, 8, 4), (8, 8, 8)],
        (16, 8): [(4, 4, 8), (2, 8, 8), (4, 4, 4), (4, 8, 4), (8, 4, 4), (8, 4, 8)],
        (8, 16): [(4, 4, 4), (4, 4, 8)],
        (16, 16): [(4, 4, 4), (2, 4, 8), (4, 2, 8), (4, 4, 8), (8, 1, 8), (8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 4)],
        ('float', 'float'): [(4, 8, 4)],
    },
    'AIE-MLV2': {
        (8, 8): [(8, 8, 8), (4, 8, 8)],
        (16, 8): [(4, 4, 8), (8, 2, 8)],
        (8, 16): [(4, 4, 8), (8, 2, 8)],
        (16, 16): [(8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 8)],
        ('float', 'float'): [(4, 8, 4)],
    },
}


def select_generation_key(generation: str) -> str:
    norm = (generation or '').upper()
    for key in sorted(TILING_OPTIONS.keys(), key=len, reverse=True):
        if key in norm:
            return key
    return 'AIE'


def tiling_key(dtype) -> Any:
    c_type = getattr(dtype, 'c_type', '') or ''
    if c_type in ('bfloat16', 'float', 'float32'):
        return c_type
    return int(dtype.width)


def storage_bytes_for_spec(spec) -> int:
    return max(1, int((spec.width + 7) // 8))


def describe_family_lhs_staging(variant, consumer, config, tensor_name, port, buf_dims=None):
    p = config.parameters
    tile_m = int(p.tiling.tile_m)
    tile_k = int(p.tiling.tile_k)
    in_slice = int(p.lhs_feat_slice)
    raw_in = int(p.lhs_slice_raw)
    view = variant._io_view(consumer, tensor_name, 'inputs')
    shapes = p.io_shapes['inputs'][tensor_name]
    buffer_order = list(view['buffer_order'])
    buffer_dimension = (
        [int(shapes['padded'][i]) for i in buffer_order] if buf_dims is None else [int(x) for x in buf_dims]
    )
    boundary_dimension = [int(shapes['logical'][i]) for i in buffer_order]
    io_boundary_dimension = list(boundary_dimension)
    io_tiling_dimension = list(io_boundary_dimension)
    feat_dim, indep_dim, traversal_dims = variant._canonical_buffer_axes(view, len(buffer_dimension), buffer_order)
    io_tiling_dimension[feat_dim] = raw_in
    tiling_dimension = [1 for _ in buffer_dimension]
    tiling_dimension[feat_dim] = tile_k
    tiling_dimension[indep_dim] = tile_m
    tile_traversal = []
    for dim in traversal_dims:
        if dim == feat_dim:
            tile_traversal.append({'dimension': feat_dim, 'stride': tile_k, 'wrap': in_slice // tile_k})
        elif dim == indep_dim:
            tile_traversal.append(
                {'dimension': indep_dim, 'stride': tile_m, 'wrap': buffer_dimension[indep_dim] // tile_m}
            )
        else:
            tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})
    offset = [0 for _ in buffer_dimension]
    offset[feat_dim] = port * in_slice
    return {
        'access': 'read',
        'buffer_dimension': buffer_dimension,
        'tiling_dimension': tiling_dimension,
        'offset': offset,
        'tile_traversal': tile_traversal,
        'boundary_dimension': boundary_dimension,
        'io_tiling_dimension': io_tiling_dimension,
        'io_boundary_dimension': io_boundary_dimension,
        'slice_dimension': feat_dim,
        'feature_dimension': feat_dim,
        'independent_dimension': indep_dim,
    }


def describe_family_output_staging(variant, node, config, tensor_name, port, buf_dims=None):
    p = config.parameters
    tile_m = int(p.tiling.tile_m)
    tile_n = int(p.tiling.tile_n)
    out_slice = int(p.rhs_feat_slice)
    raw_out = int(p.rhs_slice_raw)
    view = variant._io_view(node, tensor_name, 'outputs')
    shapes = p.io_shapes['outputs'][tensor_name]
    buffer_order = list(view['buffer_order'])
    buffer_dimension = (
        [int(shapes['padded'][i]) for i in buffer_order] if buf_dims is None else [int(x) for x in buf_dims]
    )
    io_boundary_dimension = [int(shapes['real'][i]) for i in buffer_order]
    io_tiling_dimension = list(io_boundary_dimension)
    feat_dim, indep_dim, traversal_dims = variant._canonical_buffer_axes(view, len(buffer_dimension), buffer_order)
    io_tiling_dimension[feat_dim] = raw_out
    tiling_dimension = [1 for _ in buffer_dimension]
    tiling_dimension[feat_dim] = tile_n
    tiling_dimension[indep_dim] = tile_m
    tile_traversal = []
    for dim in traversal_dims:
        if dim == feat_dim:
            tile_traversal.append({'dimension': feat_dim, 'stride': tile_n, 'wrap': out_slice // tile_n})
        elif dim == indep_dim:
            tile_traversal.append(
                {'dimension': indep_dim, 'stride': tile_m, 'wrap': buffer_dimension[indep_dim] // tile_m}
            )
        else:
            tile_traversal.append({'dimension': dim, 'stride': 1, 'wrap': buffer_dimension[dim]})
    offset = [0 for _ in buffer_dimension]
    offset[feat_dim] = port * out_slice
    return {
        'access': 'write',
        'buffer_dimension': buffer_dimension,
        'tiling_dimension': tiling_dimension,
        'offset': offset,
        'tile_traversal': tile_traversal,
        'io_tiling_dimension': io_tiling_dimension,
        'io_boundary_dimension': io_boundary_dimension,
        'slice_dimension': feat_dim,
        'feature_dimension': feat_dim,
        'independent_dimension': indep_dim,
    }


def validate_family_tile_contract(
    *,
    node_name: str,
    precision,
    parallelism,
    tiling,
    padded_independent_extent: int,
    lhs_feat_slice: int,
    rhs_feat_slice: int,
    padded_lhs_features: int,
    padded_rhs_features: int,
    bank_bytes: int = 16 * 1024,
    rhs_overhead_bytes: int = 0,
) -> None:
    a_tile_bytes = int(padded_independent_extent) * int(lhs_feat_slice) * storage_bytes_for_spec(precision['lhs'])
    b_tile_bytes = int(lhs_feat_slice) * int(rhs_feat_slice) * storage_bytes_for_spec(precision['rhs'])
    c_tile_bytes = int(padded_independent_extent) * int(rhs_feat_slice) * storage_bytes_for_spec(precision['output'])
    if a_tile_bytes > bank_bytes:
        raise ValueError(f'{node_name}: A tile uses {a_tile_bytes}B, exceeds one {bank_bytes}B bank.')
    if b_tile_bytes + int(rhs_overhead_bytes) > bank_bytes:
        raise ValueError(
            f'{node_name}: B tile plus overhead uses '
            f'{b_tile_bytes + int(rhs_overhead_bytes)}B, exceeds one {bank_bytes}B bank.'
        )
    if c_tile_bytes > bank_bytes:
        raise ValueError(f'{node_name}: C tile uses {c_tile_bytes}B, exceeds one {bank_bytes}B bank.')

    if int(padded_lhs_features) != int(lhs_feat_slice) * int(parallelism.cas_length):
        raise ValueError(f'{node_name}: padded_lhs_features must equal lhs_feat_slice * cas_length.')
    if int(padded_rhs_features) != int(rhs_feat_slice) * int(parallelism.cas_num):
        raise ValueError(f'{node_name}: padded_rhs_features must equal rhs_feat_slice * cas_num.')
    if int(lhs_feat_slice) % max(1, 2 * int(tiling.tile_k)) != 0:
        raise ValueError(f'{node_name}: lhs_feat_slice must be divisible by 2 * tile_k.')
    if int(rhs_feat_slice) % max(1, 2 * int(tiling.tile_n)) != 0:
        raise ValueError(f'{node_name}: rhs_feat_slice must be divisible by 2 * tile_n.')
    if int(padded_independent_extent) % max(1, 2 * int(tiling.tile_m)) != 0:
        raise ValueError(f'{node_name}: padded_independent_extent must be divisible by 2 * tile_m.')


def np_dtype_for_spec(spec) -> np.dtype:
    c_type = getattr(spec, 'c_type', '') or ''
    if c_type == 'bfloat16':
        return np.uint16
    if c_type in ('float', 'float32'):
        return np.float32
    return np.int8 if int(spec.width) <= 8 else np.int16


def np_bias_dtype_for_spec(spec) -> np.dtype:
    c_type = getattr(spec, 'c_type', '') or ''
    if c_type in ('bfloat16', 'float', 'float32'):
        return np.float32
    return np.int16 if int(spec.width) <= 16 else np.int32


def pack_as_float(array: np.ndarray, fmt: FloatFormat) -> np.ndarray:
    """Cast weight/bias data to the float storage format required by mmul kernels."""
    if array is None:
        return None
    if fmt == FloatFormat.BF16:
        f32 = np.asarray(array, dtype=np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    return np.asarray(array, dtype=np.float32)


def quantize_to_int(
    array: np.ndarray,
    frac_bits: int,
    target_bits: int,
    signed: bool = True,
    rounding_mode=None,
    saturation_mode=None,
) -> np.ndarray:
    """Quantize float weight/bias data to fixed-point integers for mmul kernels."""
    if array is None:
        return None
    scale = 1 << frac_bits if frac_bits > 0 else 1
    scaled = np.asarray(array, dtype=np.float64) * scale
    rounded = apply_rounding(scaled, rounding_mode)
    integers = rounded.astype(np.int64)
    processed = handle_overflow(integers, target_bits, signed, saturation_mode)
    dtype = dtype_for_precision(target_bits, signed)
    return processed.astype(dtype, copy=False)


def pack_mmul_rhs_matrix(
    W,
    *,
    K: int,
    N: int,
    K_slice: int,
    N_slice: int,
    tile_k: int,
    tile_n: int,
    cas_length: int,
    cas_num: int,
    order: str = 'C',
    dtype=None,
):
    assert tile_k > 0 and tile_n > 0
    assert K_slice % tile_k == 0
    assert N_slice % tile_n == 0

    W = np.asarray(W)
    if dtype is not None:
        W = W.astype(dtype, copy=False)
    if W.ndim < 2:
        raise ValueError('W must have at least 2 dimensions')
    W_kn = W.reshape((-1, K, N))[-1]

    tiles_per_k = K_slice // tile_k
    tiles_per_n = N_slice // tile_n
    elements_per_tile = tile_k * tile_n
    flat_len = tiles_per_k * tiles_per_n * elements_per_tile

    packed = np.zeros((cas_num, cas_length, flat_len), dtype=W_kn.dtype)
    tile_buf = np.zeros((tile_k, tile_n), dtype=W_kn.dtype)

    for chain in range(cas_num):
        n_base = chain * N_slice
        for cas in range(cas_length):
            flat = packed[chain, cas]
            tile_idx = 0
            for k_tile in range(tiles_per_k):
                gk = cas * K_slice + k_tile * tile_k
                real_k = max(0, min(tile_k, K - gk))
                for n_tile in range(tiles_per_n):
                    tile_buf.fill(0)
                    gn = n_base + n_tile * tile_n
                    real_n = max(0, min(tile_n, N - gn))
                    if real_k > 0 and real_n > 0:
                        tile_buf[:real_k, :real_n] = W_kn[gk : gk + real_k, gn : gn + real_n]
                    start = tile_idx * elements_per_tile
                    flat[start : start + elements_per_tile] = tile_buf.ravel(order=order)
                    tile_idx += 1

    return packed


def pack_vector_by_n_slice(
    v,
    *,
    N: int,
    N_slice: int,
    cas_num: int,
    dtype=None,
):
    v = np.asarray(v)
    if dtype is not None:
        v = v.astype(dtype, copy=False)
    if v.ndim > 1:
        v = v.reshape((-1,))[:N]
    if v.shape[0] != N:
        raise ValueError(f'Vector length mismatch: got {v.shape[0]}, expected {N}')

    packed = np.zeros((cas_num, N_slice), dtype=v.dtype)
    for chain in range(cas_num):
        n_base = chain * N_slice
        real = max(0, min(N_slice, N - n_base))
        if real > 0:
            packed[chain, :real] = v[n_base : n_base + real]
    return packed
