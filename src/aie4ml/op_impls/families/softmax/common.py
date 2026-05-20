from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ...utils.precision import storage_bytes_for_spec

DEFAULT_INV_SHIFT = 15
SOFTMAX_I8_VEC_SIZE = 32


@dataclass(frozen=True)
class SoftmaxTiling:
    cas_num: int
    tile_outer: int


def softmax_vec_size(precision, device) -> int:
    if precision.format == 'int8':
        return SOFTMAX_I8_VEC_SIZE
    elem_bytes = storage_bytes_for_spec(precision)
    return int(device.vector_bytes) // max(1, int(elem_bytes))


def validate_softmax_tile_contract(
    *,
    node_name: str,
    precision: Dict[str, Any],
    rows: int,
    cols: int,
    bank_bytes: int,
    vec_size: int,
) -> None:
    if int(rows) <= 0 or int(cols) <= 0:
        raise ValueError(f'{node_name}: Softmax rows/cols must be positive, got rows={rows}, cols={cols}.')
    if int(cols) % int(vec_size) != 0:
        raise ValueError(
            f'{node_name}: softmax inner dimension {cols} must be a multiple of vec_size={vec_size}; '
            'pad the softmax axis before lowering.'
        )

    in_bytes = int(rows) * int(cols) * storage_bytes_for_spec(precision['lhs'])
    out_bytes = int(rows) * int(cols) * storage_bytes_for_spec(precision['output'])
    scratch_bytes = int(cols) * 2
    if in_bytes > bank_bytes:
        raise ValueError(f'{node_name}: softmax input tile uses {in_bytes}B, exceeds one {bank_bytes}B bank.')
    if out_bytes > bank_bytes:
        raise ValueError(f'{node_name}: softmax output tile uses {out_bytes}B, exceeds one {bank_bytes}B bank.')
    if scratch_bytes > bank_bytes:
        raise ValueError(f'{node_name}: softmax scratch row uses {scratch_bytes}B, exceeds one {bank_bytes}B bank.')


def resolve_softmax_parallelism(
    *,
    outer_prefix: int,
    last_outer: int,
    full_inner: int,
    elem_in_bytes: int,
    elem_out_bytes: int,
    device,
    requested_cas_num: int | None = None,
) -> SoftmaxTiling:
    max_rows = max(1, int(device.rows))
    bank_bytes = int(device.bank_mem_bytes)

    if requested_cas_num is not None:
        candidates = [int(requested_cas_num)]
    else:
        candidates = list(range(1, max_rows + 1))

    for cas_num in candidates:
        if cas_num < 1 or cas_num > min(max_rows, int(last_outer)):
            continue
        if int(last_outer) % int(cas_num) != 0:
            continue
        tile_outer = int(last_outer) // int(cas_num)
        rows = int(outer_prefix) * int(tile_outer)
        in_bytes = rows * int(full_inner) * max(1, int(elem_in_bytes))
        out_bytes = rows * int(full_inner) * max(1, int(elem_out_bytes))
        scratch_bytes = int(full_inner) * 2
        if in_bytes <= bank_bytes and out_bytes <= bank_bytes and scratch_bytes <= bank_bytes:
            return SoftmaxTiling(cas_num=int(cas_num), tile_outer=int(tile_outer))

    raise ValueError(
        f'No legal HCCS Softmax parallelism: last_outer={last_outer} must split into a '
        f'cas_num<={max_rows} producing per-kernel row tiles within {bank_bytes}B '
        f'(outer_prefix={outer_prefix}, full_inner={full_inner}, '
        f'in_bytes/elem={elem_in_bytes}, out_bytes/elem={elem_out_bytes}).'
    )


def _as_int_array(value: Any, *, name: str, lo: int, hi: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f'HCCS Softmax parameter {name!r} must not be empty.')
    if np.any((arr < int(lo)) | (arr > int(hi))):
        raise ValueError(f'HCCS Softmax parameter {name!r} must be in [{lo}, {hi}].')
    return arr


def infer_hccs_param_sets(hccs: Dict[str, Any]) -> int:
    explicit = hccs.get('param_sets')
    lengths = []
    for name in ('B', 'S', 'Dmax'):
        size = int(np.asarray(hccs[name]).reshape(-1).size)
        if size < 1:
            raise ValueError(f'HCCS Softmax parameter {name!r} must not be empty.')
        if size > 1:
            lengths.append(size)

    inferred = int(lengths[0]) if lengths else 1
    if any(length != inferred for length in lengths):
        raise ValueError(f'HCCS Softmax non-scalar parameter lengths must match, got {lengths}.')

    param_sets = int(explicit) if explicit is not None else inferred
    if param_sets < 1:
        raise ValueError(f'HCCS Softmax param_sets must be positive, got {param_sets}.')
    if inferred > 1 and int(param_sets) != int(inferred):
        raise ValueError(f'HCCS Softmax param_sets={param_sets} must match non-scalar parameter length {inferred}.')
    return param_sets


def _compact_hccs_param(values: np.ndarray, *, name: str, param_sets: int) -> np.ndarray:
    if values.size == 1:
        return np.full((int(param_sets),), int(values[0]), dtype=np.int64)
    if values.size == int(param_sets):
        return values.astype(np.int64, copy=True)
    raise ValueError(f'HCCS Softmax parameter {name!r} length must be 1 or param_sets={param_sets}; got {values.size}.')


def pack_hccs_params(
    hccs: Dict[str, Any],
    *,
    param_sets: int,
    cols: int,
    cas_num: int,
) -> Dict[str, np.ndarray]:
    b = _compact_hccs_param(
        _as_int_array(hccs['B'], name='B', lo=0, hi=32767),
        name='B',
        param_sets=param_sets,
    )
    s = _compact_hccs_param(
        _as_int_array(hccs['S'], name='S', lo=0, hi=127),
        name='S',
        param_sets=param_sets,
    )
    dmax = _compact_hccs_param(
        _as_int_array(hccs['Dmax'], name='Dmax', lo=0, hi=127),
        name='Dmax',
        param_sets=param_sets,
    )

    min_score = b - s * dmax
    if np.any(min_score < 0):
        bad = int(np.argmin(min_score))
        raise ValueError(
            f'HCCS Softmax requires B - S*Dmax >= 0 for every row; ' f'row {bad} gives {int(min_score[bad])}.'
        )
    if np.any(min_score * int(cols) < 256):
        bad = int(np.argmin(min_score))
        raise ValueError(
            f'HCCS Softmax requires cols * (B - S*Dmax) >= 256 for reciprocal range; '
            f'row {bad} gives {int(min_score[bad]) * int(cols)}.'
        )
    if np.any(b * int(cols) > 32767):
        raise ValueError('HCCS Softmax requires cols * B <= 32767 so score sum fits int16 range.')

    if int(param_sets) != 1:
        raise ValueError(f'HCCS Softmax compile-time scalar parameter packing requires param_sets=1, got {param_sets}.')

    packed_b = np.full((int(cas_num),), int(b[0]), dtype=np.int16)
    packed_s = np.full((int(cas_num),), int(s[0]), dtype=np.int8)
    packed_dmax = np.full((int(cas_num),), int(dmax[0]), dtype=np.uint8)

    return {'B': packed_b, 'S': packed_s, 'Dmax': packed_dmax}
