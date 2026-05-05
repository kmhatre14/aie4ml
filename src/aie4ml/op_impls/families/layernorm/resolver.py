from __future__ import annotations

import math
from dataclasses import dataclass

from ....aie_types import AIEDataType, FloatIntent
from ....ir import input_tensor_for_role
from ...family_registry import family_resolver
from ...registry import select_variant as _select_variant
from ...utils import build_tensor_view
from ...utils.io import view_shape
from ...utils.precision import (
    aie_rounding_token,
    resolve_exact_storage_dtype,
    storage_bytes_for_spec,
    to_quant_intent,
)
from .common import (
    BETA_FRAC_BITS,
    DEFAULT_ISQRT_NR_ITERS,
    DEFAULT_USE_AIE_INVSQRT,
    GAMMA_FRAC_BITS,
    layernorm_vec_size,
)
from .config import LayerNormConfig, LayerNormParallelismConfig


@dataclass(frozen=True)
class LayerNormTiling:
    cas_num: int
    tile_outer: int


def _resolve_eps_q0(node_name: str, metadata: dict, input_frac: int) -> int:
    if 'epsilon' in metadata:
        epsilon = float(metadata['epsilon'])
        if epsilon < 0.0:
            raise ValueError(f'{node_name}: LayerNormalization epsilon must be non-negative, got {epsilon}.')
        eps_q0_f = epsilon * float(1 << (2 * int(input_frac)))
        eps_q0 = int(round(eps_q0_f))
        if not math.isclose(eps_q0_f, float(eps_q0), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f'{node_name}: epsilon={epsilon} is not exactly representable as integer EPS_Q0 '
                f'for input frac={input_frac} (epsilon / input_scale^2 = {eps_q0_f}).'
            )
        if eps_q0 < 1:
            raise ValueError(
                f'{node_name}: epsilon={epsilon} maps to EPS_Q0={eps_q0}; '
                'integer LayerNorm requires a positive variance floor.'
            )
        return eps_q0

    eps_q0 = int(metadata.get('eps_q0', 1))
    if eps_q0 < 1:
        raise ValueError(f'{node_name}: eps_q0 must be positive, got {eps_q0}.')
    return eps_q0


def resolve_layernorm_parallelism(
    *,
    outer_prefix: int,
    last_outer: int,
    full_inner: int,
    elem_in_bytes: int,
    elem_out_bytes: int,
    gamma_bytes: int,
    beta_bytes: int,
    device,
    requested_cas_num: int | None = None,
) -> LayerNormTiling:
    """Pick cas_num such that each kernel owns a whole-row outer tile that fits one bank.

    The kernel needs the complete full_inner extent to compute mean/variance,
    so only the last outer axis is partitioned. Any leading outer axes are
    preserved here and later folded into the 2-D hardware contract by
    CompactBufferRank. cas_num must divide last_outer exactly.
    """
    max_rows = max(1, int(device.rows))
    bank_bytes = int(device.bank_mem_bytes)

    if full_inner * gamma_bytes > bank_bytes or full_inner * beta_bytes > bank_bytes:
        raise ValueError(
            f'full_inner={full_inner} parameter buffers exceed one {bank_bytes}B bank '
            f'(gamma {full_inner * gamma_bytes}B, beta {full_inner * beta_bytes}B).'
        )

    if requested_cas_num is not None:
        candidates = [int(requested_cas_num)]
    else:
        candidates = list(range(max_rows, 0, -1))

    for cas_num in candidates:
        if cas_num < 1 or cas_num > min(max_rows, int(last_outer)):
            continue
        if last_outer % cas_num != 0:
            continue
        tile_outer = int(last_outer) // int(cas_num)
        compacted_tile_outer = int(outer_prefix) * int(tile_outer)
        in_bytes = compacted_tile_outer * full_inner * max(1, int(elem_in_bytes))
        out_bytes = compacted_tile_outer * full_inner * max(1, int(elem_out_bytes))
        if in_bytes <= bank_bytes and out_bytes <= bank_bytes:
            return LayerNormTiling(cas_num=int(cas_num), tile_outer=int(tile_outer))

    raise ValueError(
        f'No legal LayerNorm parallelism: last_outer={last_outer} must split into a '
        f'cas_num<={max_rows} producing per-kernel tiles within {bank_bytes}B '
        f'(outer_prefix={outer_prefix}, full_inner={full_inner}, '
        f'in_bytes/elem={elem_in_bytes}, out_bytes/elem={elem_out_bytes}).'
    )


@family_resolver('layer_norm')
class LayerNormResolver:
    op_type = 'layer_norm'

    def resolve(self, node, device, directives=None) -> LayerNormConfig:
        io_route = dict((directives or {}).get('io_route', {}))
        input_contracts = (directives or {}).get('input_contracts', {})

        in_tensor = input_tensor_for_role(node, 'lhs')
        input_tensor_for_role(node, 'gamma')
        input_tensor_for_role(node, 'beta')
        out_tensor = node.outputs[0]

        if isinstance(in_tensor.precision, FloatIntent) or isinstance(out_tensor.precision, FloatIntent):
            raise ValueError(f'{node.name}: integer LayerNorm requires int8 input/output precision.')

        precision = {
            'lhs': resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
            'gamma': AIEDataType(format='int16', frac=GAMMA_FRAC_BITS),
            'beta': AIEDataType(format='int16', frac=BETA_FRAC_BITS),
        }
        if (
            int(precision['lhs'].width) != 8
            or int(precision['output'].width) != 8
            or not bool(precision['lhs'].signed)
            or not bool(precision['output'].signed)
        ):
            raise ValueError(
                f'{node.name}: integer LayerNorm requires signed int8 input and output, '
                f"got input={precision['lhs'].format!r}, output={precision['output'].format!r}."
            )

        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        if len(in_shape) < 2:
            raise ValueError(f'{node.name}: LayerNorm requires rank>=2 input tensors, got {len(in_shape)}.')

        full_inner = int(in_shape[-1])
        outer_prefix = int(math.prod(in_shape[:-2])) if len(in_shape) > 2 else 1
        last_outer = int(in_shape[-2])

        vec_size = layernorm_vec_size(precision['lhs'], device)
        if full_inner % vec_size != 0:
            raise ValueError(
                f'{node.name}: full_inner={full_inner} must be a multiple of vec_size={vec_size}; '
                'pad the inner dimension before LayerNorm.'
            )
        if full_inner <= 0 or (full_inner & (full_inner - 1)) != 0:
            raise ValueError(
                f'{node.name}: full_inner={full_inner} must be a power of two for the integer LayerNorm kernel.'
            )

        in_contract = input_contracts.get(in_tensor.name)
        parallel_cfg = (directives or {}).get('parallelism', {}) or {}
        cas_length = int(parallel_cfg.get('cas_length', 1))
        if cas_length != 1:
            raise ValueError(f'{node.name}: integer LayerNorm requires cas_length=1, got {cas_length}.')
        # Prefer the user override; otherwise reuse producer's cas_num when its
        # contract matches ours so direct transport stays legal. With a mismatched
        # producer contract (e.g. Dense 'inner') transport_classify will insert a
        # memtile automatically and cas_num is free.
        if parallel_cfg.get('cas_num') is not None:
            requested_cas_num = int(parallel_cfg['cas_num'])
        elif in_contract is not None and in_contract.contract == 'outer':
            requested_cas_num = len(in_contract.port_staging)
        else:
            requested_cas_num = None

        tiling = resolve_layernorm_parallelism(
            outer_prefix=outer_prefix,
            last_outer=last_outer,
            full_inner=full_inner,
            elem_in_bytes=storage_bytes_for_spec(precision['lhs']),
            elem_out_bytes=storage_bytes_for_spec(precision['output']),
            gamma_bytes=storage_bytes_for_spec(precision['gamma']),
            beta_bytes=storage_bytes_for_spec(precision['beta']),
            device=device,
            requested_cas_num=requested_cas_num,
        )

        tile_outer = tiling.tile_outer
        compacted_tile_outer = int(outer_prefix) * int(tile_outer)
        tile_inner = full_inner

        io_views = {}
        io_views[in_tensor.name] = build_tensor_view(
            node,
            in_tensor,
            'inputs',
            full_inner=full_inner,
            tile_inner=tile_inner,
            tile_inner_raw=tile_inner,
            full_outer=last_outer,
            tile_outer=tile_outer,
            tile_outer_raw=tile_outer,
        )
        io_views[out_tensor.name] = build_tensor_view(
            node,
            out_tensor,
            'outputs',
            full_inner=full_inner,
            tile_inner=tile_inner,
            tile_inner_raw=tile_inner,
            full_outer=last_outer,
            tile_outer=tile_outer,
            tile_outer_raw=tile_outer,
        )

        out_intent = to_quant_intent(out_tensor.precision)
        out_shift = int(out_intent.frac)
        if out_shift < 0 or out_shift > 15:
            raise ValueError(
                f'{node.name}: output frac={out_shift} must be in [0, 15]; '
                'integer LayerNorm cannot left-shift the accumulator or exceed NORM_SHIFT=15.'
            )

        rounding_mode = aie_rounding_token(precision['output'])
        eps_q0 = _resolve_eps_q0(node.name, node.metadata, int(precision['lhs'].frac))

        return LayerNormConfig(
            precision=precision,
            parallelism=LayerNormParallelismConfig(cas_num=int(tiling.cas_num), cas_length=cas_length),
            rows=int(compacted_tile_outer),
            cols=int(full_inner),
            vec_size=int(vec_size),
            gamma_shift=int(GAMMA_FRAC_BITS),
            out_shift=int(out_shift),
            eps_q0=int(eps_q0),
            isqrt_nr_iters=int(DEFAULT_ISQRT_NR_ITERS),
            use_aie_invsqrt=bool(DEFAULT_USE_AIE_INVSQRT),
            rounding_mode=rounding_mode,
            io_views=io_views,
            io_route=io_route,
        )

    def select_variant(self, config: LayerNormConfig, generation: str):
        return _select_variant(self.op_type, config, generation)
