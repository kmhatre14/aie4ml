# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Policy-driven resolver registry for AIE attribute resolution."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..aie_types import AIEDataType, FloatFormat, FloatIntent, QuantIntent, RoundingMode, SaturationMode
from ..ir import ResolvedAttributes, TraitInstance
from ..op_impls import get_op_impl_registry
from ..op_impls.families.matmul import tiling_key

log = logging.getLogger(__name__)


ResolverFn = Callable[['LayerResolveContext'], None]


@dataclass(frozen=True)
class LayerPolicy:
    """Describes the resolver pipeline for a layer class."""

    namespaces: Tuple[str, ...]
    resolvers: Tuple[ResolverFn, ...]
    requires_numeric: bool = False
    op_family: Optional[str] = None
    static_parameter_roles: frozenset[str] = frozenset()


@dataclass
class LayerResolveContext:
    """State shared across resolver functions."""

    backend_ctx: Any
    node: Any
    layer_name: str
    layer_class: str
    policy: LayerPolicy
    quant: Dict[str, Any]
    device: Any
    attributes: ResolvedAttributes
    state: Dict[str, Any] = field(default_factory=dict)

    def numeric(self) -> Optional['NumericBundle']:
        return self.state.get('numeric')

    def set_numeric(self, numeric: 'NumericBundle') -> None:
        self.state['numeric'] = numeric

    def set_parallelism(self, parallelism: 'ParallelismResult') -> None:
        self.state['parallelism'] = parallelism

    def parallelism(self) -> Optional['ParallelismResult']:
        return self.state.get('parallelism')


@dataclass
class NumericBundle:
    """Collection of resolved numeric precisions."""

    dtypes: Dict[str, AIEDataType]

    def get(self, key: str) -> Optional[AIEDataType]:
        return self.dtypes.get(key)

    def items(self):
        return self.dtypes.items()

    def to_attribute_map(self) -> Dict[str, AIEDataType]:
        filtered: Dict[str, AIEDataType] = {}
        for key, dtype in self.dtypes.items():
            if dtype is None:
                continue
            width = int(getattr(dtype, 'width', 0) or 0)
            if width > 0:
                filtered[key] = dtype
        return filtered


@dataclass
class ParallelismResult:
    """Outcome of the parallelism resolver."""

    cas_num: int = 1
    cas_length: int = 1
    lhs_slice: int = 0
    rhs_slice: int = 0
    lhs_slice_raw: int = 0
    rhs_slice_raw: int = 0
    lhs_tile_bytes: int = 0
    rhs_tile_bytes: int = 0
    output_tile_bytes: int = 0
    parallel_factor: int = 1
    lhs_alignment: int = 1
    rhs_alignment: int = 1
    padded_independent_extent: int = 1
    independent_extent: int = 1
    padded_lhs_features: int = 0
    padded_rhs_features: int = 0


ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    'lhs': ('lhs',),
    'rhs': ('rhs',),
    'bias': ('bias',),
}


OP_IMPL_REGISTRY = get_op_impl_registry()

ACC_TAG_WIDTHS = {
    'acc32': 32,
    'acc48': 48,
    'acc64': 64,
}

ROUNDING_TOKEN_MAP: Dict[RoundingMode, str] = {
    RoundingMode.TRN: 'floor',
    RoundingMode.RND_MIN_INF: 'floor',
    RoundingMode.RND_INF: 'ceil',
    RoundingMode.RND: 'symmetric_inf',
    RoundingMode.TRN_ZERO: 'symmetric_zero',
    RoundingMode.RND_ZERO: 'symmetric_zero',
    RoundingMode.RND_CONV: 'conv_even',
}


def _normalize_precision_name(name: str) -> str:
    for suffix in ('_precision', '_dtype'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _input_role_map(node) -> Dict[str, str]:
    roles = list(node.metadata.get('input_roles') or [])
    return {tensor.name: str(roles[index]) for index, tensor in enumerate(node.inputs) if index < len(roles)}


def _input_tensor_for_roles(ctx: LayerResolveContext, *roles: str):
    wanted = set(roles)
    role_map = _input_role_map(ctx.node)
    for tensor in ctx.node.inputs:
        if role_map.get(tensor.name) in wanted:
            return tensor
    return None


def _input_role(node, tensor_name: str) -> str:
    return _input_role_map(node).get(tensor_name, '')


def _ctype_for_width(width: int, signed: bool) -> str:
    width = max(1, int(width))
    if signed:
        if width <= 8:
            return 'int8_t'
        if width <= 16:
            return 'int16_t'
        if width <= 32:
            return 'int32_t'
        return 'int64_t'
    if width <= 8:
        return 'uint8_t'
    if width <= 16:
        return 'uint16_t'
    if width <= 32:
        return 'uint32_t'
    return 'uint64_t'


def _ctype_for_float(fmt: FloatFormat) -> str:
    if fmt == FloatFormat.BF16:
        return 'bfloat16'
    return 'float'


def _to_quant_intent(precision: Any) -> QuantIntent:
    if isinstance(precision, QuantIntent):
        return precision
    if isinstance(precision, AIEDataType):
        return QuantIntent(
            width=int(precision.width),
            frac=int(precision.frac),
            signed=bool(precision.signed),
            rounding=precision.rounding,
            saturation=precision.saturation,
        )
    raise TypeError(f'Unsupported precision representation {type(precision)}')


def _resolve_storage_width(width: int, *, allowed: Tuple[int, ...], namespace: str, layer_name: str) -> int:
    width = int(width)
    if width <= 0:
        raise ValueError(f'{layer_name}: invalid {namespace} width {width}')
    for candidate in allowed:
        if width <= candidate:
            return candidate
    raise ValueError(f'{layer_name}: {namespace} width {width} exceeds supported widths {allowed}')


def _resolve_storage_dtype(
    intent: QuantIntent, *, allowed: Tuple[int, ...], namespace: str, layer_name: str
) -> AIEDataType:
    storage_width = _resolve_storage_width(intent.width, allowed=allowed, namespace=namespace, layer_name=layer_name)
    return AIEDataType(
        width=storage_width,
        signed=bool(intent.signed),
        frac=int(intent.frac),
        rounding=intent.rounding,
        saturation=intent.saturation,
        c_type=_ctype_for_width(storage_width, bool(intent.signed)),
    )


def _resolve_numeric_float(ctx: LayerResolveContext) -> None:
    """Float path for resolve_numeric: no shift, no accumulator tag, c_type from FloatFormat."""
    resolved: Dict[str, AIEDataType] = {}
    allowed = {'lhs', 'output'}
    if ctx.policy.requires_numeric:
        allowed.add('rhs')
        if 'bias' in ctx.policy.static_parameter_roles:
            allowed.add('bias')
    for key in allowed:
        prec = ctx.quant.get(f'{key}_precision')
        if prec is None:
            continue
        if isinstance(prec, FloatIntent):
            resolved[key] = AIEDataType(
                width=prec.width,
                signed=True,
                frac=0,
                c_type=_ctype_for_float(prec.format),
            )
    if ctx.policy.requires_numeric:
        if 'bias' in ctx.policy.static_parameter_roles:
            resolved['bias'] = AIEDataType(width=32, signed=True, frac=0, c_type='float')
        ctx.attributes.scalars['shift'] = 0
        ctx.attributes.scalars['accumulator_tag'] = 'accfloat'
        ctx.attributes.scalars['rounding_mode'] = 'conv_even'
        resolved.setdefault('acc', AIEDataType(width=32, signed=True, frac=0, c_type='accfloat'))
    numeric = NumericBundle(resolved)
    ctx.attributes.numeric.update(numeric.to_attribute_map())
    ctx.set_numeric(numeric)


def resolve_numeric(ctx: LayerResolveContext) -> None:
    """Resolve backend storage types and post-shift from quantization intent."""

    # Float path: any precision that is a FloatIntent takes the float branch.
    lhs_prec = ctx.quant.get('lhs_precision')
    if isinstance(lhs_prec, FloatIntent):
        _resolve_numeric_float(ctx)
        return

    precision_entries: Dict[str, Any] = {}
    for key, value in ctx.quant.items():
        if key.endswith('_precision'):
            if value is None:
                raise RuntimeError(f'{ctx.layer_name}: quant metadata "{key}" is None')
            precision_entries[key] = value

    allowed = {'lhs', 'output'}
    if ctx.policy.requires_numeric:
        allowed |= {'rhs', 'acc'}
        if 'bias' in ctx.policy.static_parameter_roles:
            allowed.add('bias')

    required: List[str] = []
    if ctx.policy.requires_numeric:
        required.extend(['lhs', 'output', 'rhs'])
        if ctx.node.metadata.get('use_bias') and 'bias' in ctx.policy.static_parameter_roles:
            required.append('bias')

    missing = []
    for name in required:
        k = f'{name}_precision'
        if k not in precision_entries:
            missing.append(k)
    if missing:
        raise RuntimeError(f'{ctx.layer_name}: missing quant intent {", ".join(sorted(missing))}')

    intents: Dict[str, QuantIntent] = {}
    for precision_key, precision in precision_entries.items():
        alias = _normalize_precision_name(precision_key)
        if alias not in allowed:
            continue
        intents[alias] = _to_quant_intent(precision)

    resolved: Dict[str, AIEDataType] = {}

    if 'lhs' in intents:
        resolved['lhs'] = _resolve_storage_dtype(
            intents['lhs'],
            allowed=(4, 8, 16, 32),
            namespace='lhs',
            layer_name=ctx.layer_name,
        )
    if 'output' in intents:
        resolved['output'] = _resolve_storage_dtype(
            intents['output'],
            allowed=(4, 8, 16, 32),
            namespace='output',
            layer_name=ctx.layer_name,
        )

    if ctx.policy.requires_numeric:
        resolved['rhs'] = _resolve_storage_dtype(
            intents['rhs'],
            allowed=(4, 8, 16, 32),
            namespace='rhs',
            layer_name=ctx.layer_name,
        )

        if 'bias' in ctx.policy.static_parameter_roles:
            if 'bias' in intents:
                # NOTE: keep bias storage fixed to 32 bits for now;
                bias_width = 32
                resolved['bias'] = AIEDataType(
                    width=bias_width,
                    signed=bool(intents['bias'].signed),
                    # Bias is consumed in accumulator scale by the AIE dense kernels.
                    frac=int(intents['lhs'].frac + intents['rhs'].frac),
                    rounding=intents['bias'].rounding,
                    saturation=intents['bias'].saturation,
                    c_type=_ctype_for_width(bias_width, bool(intents['bias'].signed)),
                )
            else:
                bias_width = 32
                accum_frac = int(intents['lhs'].frac + intents['rhs'].frac)
                resolved['bias'] = AIEDataType(
                    width=bias_width,
                    signed=True,
                    frac=accum_frac,
                    rounding=RoundingMode.TRN,
                    saturation=SaturationMode.SAT,
                    c_type=_ctype_for_width(bias_width, True),
                )

        in_width = int(resolved['lhs'].width)
        w_width = int(resolved['rhs'].width)
        if in_width <= 8 and w_width > 8:
            raise RuntimeError(
                f'{ctx.layer_name}: unsupported int8 x int16 precision mix for AIE implementations; '
                'no implementation variant available.'
            )

        shift = int(intents['lhs'].frac + intents['rhs'].frac - intents['output'].frac)
        if shift < 0:
            log.warning(
                'Layer %s: computed shift=%d (requires left-shift) but negative shifts are unsafe on AIE-ML/XDNA; '
                'forcing shift=0. Bit-exactness will be lost. Consider increasing output fractional bits or '
                'reducing accumulator fractional depth so output_frac ≤ accum_frac.',
                ctx.layer_name,
                shift,
            )
            shift = 0
        ctx.attributes.scalars['shift'] = shift

        acc_tag = _infer_accumulator_tag(ctx.device, resolved['lhs'], resolved['rhs'], None)
        ctx.attributes.scalars['accumulator_tag'] = acc_tag
        ctx.attributes.scalars['rounding_mode'] = _aie_rounding_token(resolved['output'])

        if 'acc' not in resolved:
            acc_width = ACC_TAG_WIDTHS[acc_tag]
            resolved['acc'] = AIEDataType(
                width=acc_width,
                signed=True,
                frac=int(intents['lhs'].frac + intents['rhs'].frac),
                rounding=RoundingMode.TRN,
                saturation=SaturationMode.SAT,
                c_type=_ctype_for_width(acc_width, True),
            )

    numeric = NumericBundle(resolved)
    ctx.attributes.numeric.update(numeric.to_attribute_map())
    ctx.set_numeric(numeric)


def _supported_tile_options(op_family: str, gen: str, in_key, w_key):
    return OP_IMPL_REGISTRY.supported_tilings(op_family, gen, (in_key, w_key))


def _extract_tile_cfg(directives: Dict[str, Any]) -> Dict[str, int]:
    tiling = directives.get('tiling', {}) or {}
    return {key: int(tiling[key]) if key in tiling else 0 for key in ('tile_m', 'tile_n', 'tile_k')}


def _resolve_tile_cfg(
    layer_name: str,
    op_family: str,
    user_cfg: Dict[str, Any],
    device: Any,
    input_dtype: Optional[AIEDataType],
    weight_dtype: Optional[AIEDataType],
) -> Dict[str, int]:
    raw = _extract_tile_cfg(user_cfg)
    in_key = tiling_key(input_dtype) if input_dtype else 0
    w_key = tiling_key(weight_dtype) if weight_dtype else 0
    generation = getattr(device, 'generation', '') or ''

    options = _supported_tile_options(op_family, generation, in_key, w_key)
    if not options:
        raise ValueError(
            f'{layer_name}: no supported tile configs are registered for Generation={generation} and '
            f'(input={in_key!r}, weight={w_key!r}); cannot validate user tiling.'
        )

    user_specified = (raw['tile_m'] > 0) and (raw['tile_k'] > 0) and (raw['tile_n'] > 0)
    if user_specified:
        candidate = (raw['tile_m'], raw['tile_k'], raw['tile_n'])
        if candidate not in options:
            raise ValueError(
                f'{layer_name}: tiling {candidate} not supported for Generation={generation} and '
                f'(input={in_key!r}, weight={w_key!r}). Allowed: {options}'
            )
        return {'tile_m': candidate[0], 'tile_k': candidate[1], 'tile_n': candidate[2]}

    default_m, default_k, default_n = options[0]
    return {'tile_m': default_m, 'tile_k': default_k, 'tile_n': default_n}


def resolve_tiling(ctx: LayerResolveContext) -> None:
    numeric = ctx.numeric()
    if numeric is None:
        return
    lhs_dtype = numeric.get('lhs')
    rhs_dtype = numeric.get('rhs')
    if lhs_dtype is None or rhs_dtype is None:
        return

    op_family = ctx.policy.op_family or ctx.node.op_type
    tile_cfg = _resolve_tile_cfg(ctx.layer_name, op_family, ctx.node.directives, ctx.device, lhs_dtype, rhs_dtype)
    ctx.attributes.tiling.update(tile_cfg)
    ctx.state['tile_cfg'] = tile_cfg


def _device_lane_bytes(device: Any) -> int:
    norm = (getattr(device, 'generation', '') or '').upper()
    if any(token in norm for token in ('AIE-ML', 'AIE-MLV2', 'MLV2', 'XDNA', 'AIE2')):
        return 16  # NOTE this should be 32 in practice but compiler doesn't complain
    return 16


def _element_bytes(dtype: Optional[AIEDataType]) -> int:
    """Return number of bytes needed to store one element of this dtype."""
    if not dtype or not getattr(dtype, 'width', None):
        return 1

    width = int(dtype.width)
    return max(1, math.ceil(width / 8))


def _features_from_bytes(byte_alignment: int, element_bytes: int) -> int:
    if byte_alignment <= 0:
        return 1
    element_bytes = max(1, element_bytes)
    return max(1, math.ceil(byte_alignment / element_bytes))


def _lcm(a: int, b: int) -> int:
    if a <= 0:
        return max(1, b)
    if b <= 0:
        return max(1, a)
    return abs(a * b) // math.gcd(a, b)


def _lcm_many(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = _lcm(result, int(value))
    return result


def _family_alignment_rules(op_family: str, tile_m: int, tile_k: int, tile_n: int) -> Dict[str, int]:
    if op_family in ('dense', 'matmul'):
        return {
            'independent': max(1, 2 * max(1, tile_m)),
            'lhs': max(1, 2 * max(1, tile_k)),
            'rhs': max(1, 2 * max(1, tile_n)),
        }
    raise ValueError(f'Unsupported op_family {op_family!r} for alignment rule resolution.')


def _lhs_slice_alignment(device: Any, lhs_granularity: int, element_bytes: int) -> int:
    base = max(1, int(lhs_granularity))
    lane = _features_from_bytes(_device_lane_bytes(device), element_bytes)
    plio = _features_from_bytes(4, element_bytes)
    return _lcm_many([base, lane, plio])


def _rhs_slice_alignment(device: Any, rhs_granularity: int, element_bytes: int) -> int:
    base = max(1, int(rhs_granularity))
    lane = _features_from_bytes(_device_lane_bytes(device), element_bytes)
    plio = _features_from_bytes(4, element_bytes)
    return _lcm_many([base, lane, plio])


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return max(0, value)
    return ((int(value) + multiple - 1) // multiple) * multiple


def _bank_capacity_bytes(device: Any) -> int:
    return max(1, int(getattr(device, 'bank_mem_bytes', 0) or 1))


def _rhs_stack_overhead_bytes(op_family: str) -> int:
    return 1024 if op_family == 'matmul' else 0


def _tile_bank_usage(
    *,
    op_family: str,
    device: Any,
    padded_independent_extent: int,
    lhs_slice: int,
    rhs_slice: int,
    lhs_bytes: int,
    rhs_bytes: int,
    output_bytes: int,
) -> Dict[str, int]:
    lhs_tile_bytes = int(padded_independent_extent) * int(lhs_slice) * max(1, int(lhs_bytes))
    rhs_tile_bytes = int(lhs_slice) * int(rhs_slice) * max(1, int(rhs_bytes))
    rhs_tile_bytes += _rhs_stack_overhead_bytes(op_family)
    output_tile_bytes = int(padded_independent_extent) * int(rhs_slice) * max(1, int(output_bytes))
    return {
        'lhs_tile_bytes': lhs_tile_bytes,
        'rhs_tile_bytes': rhs_tile_bytes,
        'output_tile_bytes': output_tile_bytes,
        'max_bank_tile_bytes': max(lhs_tile_bytes, rhs_tile_bytes, output_tile_bytes),
        'bank_capacity_bytes': _bank_capacity_bytes(device),
    }


def _effective_view_shape(
    ctx: LayerResolveContext,
    tensor: Any,
    direction: str,
) -> List[int]:
    logical = [int(x) for x in tensor.shape]
    io_trait = ctx.node.traits.get('io_view')
    if io_trait is None:
        raise ValueError(f'{ctx.layer_name}: missing io_view trait.')
    view = io_trait.data[direction][tensor.name]
    perm = view.get('perm')
    if perm is None:
        return logical
    if sorted(perm) != list(range(len(logical))):
        raise ValueError(f'{ctx.layer_name}: invalid io_view perm {perm} for rank {len(logical)}.')
    return [int(logical[i]) for i in perm]


def _aligned_batch_size(batch: int, independent_granularity: int) -> int:
    return _align_up(int(batch), max(1, int(independent_granularity)))


def _independent_extent(ctx: LayerResolveContext) -> Tuple[int, int]:
    tensor = _input_tensor_for_roles(ctx, *ROLE_ALIASES['lhs'])
    if tensor is None:
        tensor = next((value for value in ctx.node.inputs if not value.is_parameter), None)
    if tensor is None:
        raise ValueError(f'{ctx.layer_name}: missing dynamic input tensor for independent extent resolution.')
    logical = _effective_view_shape(ctx, tensor, 'inputs')
    if len(logical) < 2:
        raise ValueError(f'{ctx.layer_name}: tensor rank must be >=2, got {len(logical)}.')
    independent = logical[:-1]
    extent = int(math.prod(independent))
    last_indep = int(independent[-1])
    return extent, last_indep


def _pad_logical_shape(
    logical: List[int],
    padded_feat: int,
    independent_granularity: int,
) -> List[int]:
    padded = list(logical)
    feature_axis = len(logical) - 1
    padded[feature_axis] = int(padded_feat)
    if feature_axis > 0:
        last_axis = feature_axis - 1
        padded[last_axis] = _align_up(int(padded[last_axis]), max(1, int(independent_granularity)))
    return padded


def _input_padded_features(ctx: LayerResolveContext, tensor) -> Optional[int]:
    role = _input_role(ctx.node, tensor.name)
    if tensor.is_parameter and role in ctx.policy.static_parameter_roles:
        return None
    if role in ROLE_ALIASES['rhs']:
        return int(ctx.attributes.scalars['padded_rhs_features'])
    return int(ctx.attributes.scalars['padded_lhs_features'])


def _resolve_io_shapes(ctx: LayerResolveContext) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    io_trait = ctx.node.traits.get('io_view')
    views = io_trait.data if io_trait else {'inputs': {}, 'outputs': {}}
    tile_m = ctx.attributes.tiling['tile_m']
    tile_k = ctx.attributes.tiling['tile_k']
    tile_n = ctx.attributes.tiling['tile_n']
    rules = _family_alignment_rules(ctx.policy.op_family or ctx.node.op_type, tile_m, tile_k, tile_n)
    shapes: Dict[str, Dict[str, Dict[str, List[int]]]] = {'inputs': {}, 'outputs': {}}

    def _view_shape(logical: List[int], view: Dict[str, Any]) -> List[int]:
        perm = view.get('perm')
        if perm is None:
            return list(logical)
        if sorted(perm) != list(range(len(logical))):
            raise ValueError(f'{ctx.layer_name}: invalid io_view perm {perm} for rank {len(logical)}.')
        return [int(logical[i]) for i in perm]

    for t in ctx.node.inputs:
        view = views['inputs'][t.name]
        logical = [int(x) for x in t.shape]
        real = _view_shape(logical, view)
        padded_feat = _input_padded_features(ctx, t)
        padded = (
            list(real)
            if padded_feat is None
            else _pad_logical_shape(real, padded_feat=padded_feat, independent_granularity=rules['independent'])
        )
        shapes['inputs'][t.name] = {'logical': logical, 'real': real, 'padded': padded}

    for t in ctx.node.outputs:
        view = views['outputs'][t.name]
        logical = [int(x) for x in t.shape]
        real = _view_shape(logical, view)
        padded = _pad_logical_shape(
            real,
            padded_feat=int(ctx.attributes.scalars['padded_rhs_features']),
            independent_granularity=rules['independent'],
        )
        shapes['outputs'][t.name] = {'logical': logical, 'real': real, 'padded': padded}

    return shapes


def _aligned_lhs_features(
    in_feat: int,
    cas_length: int,
    lhs_granularity: int,
    device: Any,
    element_bytes: int,
) -> int:
    slice_alignment = _lhs_slice_alignment(device, lhs_granularity, element_bytes)
    block = max(1, int(cas_length) * slice_alignment)
    return _align_up(int(in_feat), block)


def _validate_parallel_override(
    layer_name: str,
    op_family: str,
    chains: int,
    cas: int,
    n_in: int,
    n_out: int,
    align_k: int,
    align_n: int,
    lhs_bytes: int,
    rhs_bytes: int,
    output_bytes: int,
    padded_independent_extent: int,
    device: Any,
    allow_failure: bool = False,
) -> Optional[Dict[str, Any]]:
    if chains <= 0 or cas <= 0:
        raise ValueError(f'{layer_name}: cas_num and cas_length must be positive.')

    out_slice_raw = (n_out + chains - 1) // chains if chains else n_out
    in_slice_raw = (n_in + cas - 1) // cas if cas else n_in

    if in_slice_raw * lhs_bytes % 4 != 0:  # PLIO 32-bit align
        if allow_failure:
            return None
        raise ValueError(f'{layer_name}: raw IN slice not 32-bit aligned ({in_slice_raw * lhs_bytes}B).')

    out_slice = _align_up(out_slice_raw, align_n)
    in_slice = _align_up(in_slice_raw, align_k)

    bank_usage = _tile_bank_usage(
        op_family=op_family,
        device=device,
        padded_independent_extent=padded_independent_extent,
        lhs_slice=in_slice,
        rhs_slice=out_slice,
        lhs_bytes=lhs_bytes,
        rhs_bytes=rhs_bytes,
        output_bytes=output_bytes,
    )
    per_tile_limit = int(bank_usage['bank_capacity_bytes'])

    if bank_usage['max_bank_tile_bytes'] > per_tile_limit:
        if allow_failure:
            return None
        raise ValueError(
            f'{layer_name}: no valid (cas_num, cas_length) fits tile memory '
            f'(requested {chains}x{cas}, '
            f'A={bank_usage["lhs_tile_bytes"]}B, B={bank_usage["rhs_tile_bytes"]}B, '
            f'C={bank_usage["output_tile_bytes"]}B, limit={per_tile_limit}B).'
        )

    return {
        'cas_num': chains,
        'cas_length': cas,
        'lhs_slice_raw': in_slice_raw,
        'rhs_slice_raw': out_slice_raw,
        'lhs_slice': in_slice,
        'rhs_slice': out_slice,
        'balance': abs(in_slice - out_slice),
        **bank_usage,
    }


def _resolve_parallelism_numeric(
    ctx: LayerResolveContext,
    numeric: NumericBundle,
    tile_cfg: Dict[str, int],
) -> ParallelismResult:
    layer_name = ctx.layer_name
    if not ctx.node.inputs or not ctx.node.outputs:
        raise ValueError(f'{layer_name}: node is missing input or output tensors.')

    in_shape = _effective_view_shape(ctx, ctx.node.inputs[0], 'inputs')[-1]
    out_shape = _effective_view_shape(ctx, ctx.node.outputs[0], 'outputs')[-1]

    parallel_cfg = ctx.node.directives.get('parallelism', {}) or {}
    user_num_chains = parallel_cfg.get('cas_num')
    user_cas_length = parallel_cfg.get('cas_length')
    user_target = parallel_cfg.get('parallel_factor')
    target_parallel_factor = None if user_target in (None, 0, '') else int(user_target)

    def _validate_positive(name: str, value: Any) -> None:
        if value is None:
            return
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError(f'{layer_name}: {name} must be positive, got {value!r}.')

    _validate_positive('cas_num', user_num_chains)
    _validate_positive('cas_length', user_cas_length)
    _validate_positive('target_parallel_factor', target_parallel_factor)

    tile_m = int(tile_cfg['tile_m'])
    tile_n = int(tile_cfg['tile_n'])
    tile_k = int(tile_cfg['tile_k'])
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
        raise ValueError(f'{layer_name}: tiling not resolved before parallelism.')

    lhs_bytes = _element_bytes(numeric.get('lhs'))
    output_bytes = _element_bytes(numeric.get('output'))
    rhs_dtype = numeric.get('rhs')
    rhs_bytes = _element_bytes(rhs_dtype)
    op_family = ctx.policy.op_family or ctx.node.op_type

    rules = _family_alignment_rules(op_family, tile_m, tile_k, tile_n)
    lhs_align = _lhs_slice_alignment(ctx.device, rules['lhs'], lhs_bytes)
    rhs_align = _rhs_slice_alignment(ctx.device, rules['rhs'], output_bytes)

    max_out_ports = max(1, int(getattr(ctx.device, 'max_mem_out_ports', 0) or 0))
    max_in_ports = max(1, int(getattr(ctx.device, 'max_mem_in_ports', 0) or 0))

    indep_extent, last_indep = _independent_extent(ctx)
    padded_last = _aligned_batch_size(last_indep, rules['independent'])
    padded_indep = (indep_extent // max(1, last_indep)) * padded_last

    if user_num_chains and int(user_num_chains) > max_out_ports:
        log.warning(
            '%s: cas_num override %s exceeds single memtile out-ports %s; '
            'MemoryPlan will shard across multiple memtiles.',
            layer_name,
            user_num_chains,
            max_out_ports,
        )
    if user_cas_length and int(user_cas_length) > max_in_ports:
        log.warning(
            '%s: cas_length override %s exceeds single memtile in-ports %s; '
            'MemoryPlan will shard across multiple memtiles.',
            layer_name,
            user_cas_length,
            max_in_ports,
        )

    if user_num_chains and user_cas_length:
        override = _validate_parallel_override(
            layer_name,
            op_family,
            int(user_num_chains),
            int(user_cas_length),
            in_shape,
            out_shape,
            lhs_align,
            rhs_align,
            lhs_bytes,
            rhs_bytes,
            output_bytes,
            padded_indep,
            ctx.device,
        )
        if override is None:
            raise ValueError(f'{layer_name}: user-provided parallelism overrides are invalid.')
        parallel_factor = int(user_num_chains) * int(user_cas_length)
        candidate = {
            **override,
            'cas_num': int(user_num_chains),
            'cas_length': int(user_cas_length),
            'parallel_factor': parallel_factor,
        }
    else:
        chain_candidates = [int(user_num_chains)] if user_num_chains else list(range(1, max_out_ports + 1))
        cas_candidates = [int(user_cas_length)] if user_cas_length else list(range(1, max_in_ports + 1))

        best_pair = None  # (score_tuple, cand_dict)
        for cas in cas_candidates:
            for chains in chain_candidates:
                cand = _validate_parallel_override(
                    layer_name,
                    op_family,
                    chains,
                    cas,
                    in_shape,
                    out_shape,
                    lhs_align,
                    rhs_align,
                    lhs_bytes,
                    rhs_bytes,
                    output_bytes,
                    padded_indep,
                    ctx.device,
                    allow_failure=True,
                )
                if cand is None:
                    continue

                parallel_factor = chains * cas
                # --- scoring ---
                per_tile_limit = _bank_capacity_bytes(ctx.device)
                utilization_penalty = abs(
                    1.0 - (cand['max_bank_tile_bytes'] / per_tile_limit)
                )  # prefer to use the fullest per-bank tile use
                # aspect = cand['lhs_slice'] / cand['rhs_slice']
                shape_penalty = max(
                    0.0, (cand['rhs_slice'] - cand['lhs_slice']) / max(1.0, cand['lhs_slice'])
                )  # penalize OUT >> IN
                padding_waste = (cand['lhs_slice'] * cas - in_shape) + (
                    cand['rhs_slice'] * chains - out_shape
                )  # Penalize alignment padding
                match_penalty = 0 if target_parallel_factor is None else abs(parallel_factor - target_parallel_factor)
                if target_parallel_factor is not None:
                    exact_miss = int(parallel_factor != target_parallel_factor)  # 0 for exact, 1 otherwise
                    match_penalty = abs(parallel_factor - target_parallel_factor)
                    score = (
                        exact_miss,
                        match_penalty,
                        utilization_penalty,
                        shape_penalty,
                        padding_waste,
                        -parallel_factor,
                    )
                else:
                    score = (utilization_penalty, shape_penalty, padding_waste, -parallel_factor)

                if best_pair is None or score < best_pair[0]:
                    best_pair = (
                        score,
                        {
                            **cand,
                            'cas_num': chains,
                            'cas_length': cas,
                            'parallel_factor': parallel_factor,
                        },
                    )

        if best_pair is None:
            raise ValueError(
                f'{layer_name}: no valid (cas_num, cas_length) fits tile memory '
                f'(n_in={in_shape}, n_out={out_shape}, bank limit={_bank_capacity_bytes(ctx.device)}B). '
                'Try adjusting: parallelism, tiling, tensor shapes, precision, or device memory.'
            )
        candidate = best_pair[1]

    cas_num = int(candidate['cas_num'])
    cas_length = int(candidate['cas_length'])
    raw_lhs_slice = int(candidate['lhs_slice_raw'])
    raw_rhs_slice = int(candidate['rhs_slice_raw'])
    lhs_slice = int(candidate['lhs_slice'])
    rhs_slice = int(candidate['rhs_slice'])

    # (Already aligned by validator; these are no-ops but safe)
    if lhs_slice > 0:
        lhs_slice = _align_up(lhs_slice, lhs_align)
    if rhs_slice > 0:
        rhs_slice = _align_up(rhs_slice, rhs_align)

    padded_lhs = _aligned_lhs_features(in_shape, cas_length, rules['lhs'], ctx.device, lhs_bytes)
    padded_lhs = max(padded_lhs, lhs_slice * max(1, cas_length))
    padded_rhs = rhs_slice * max(1, cas_num)

    return ParallelismResult(
        cas_num=cas_num,
        cas_length=cas_length,
        lhs_slice=lhs_slice,
        rhs_slice=rhs_slice,
        lhs_slice_raw=raw_lhs_slice,
        rhs_slice_raw=raw_rhs_slice,
        lhs_tile_bytes=int(candidate['lhs_tile_bytes']),
        rhs_tile_bytes=int(candidate['rhs_tile_bytes']),
        output_tile_bytes=int(candidate['output_tile_bytes']),
        parallel_factor=int(candidate['parallel_factor']),
        lhs_alignment=lhs_align,
        rhs_alignment=rhs_align,
        padded_independent_extent=int(padded_indep),
        independent_extent=int(indep_extent),
        padded_lhs_features=int(padded_lhs),
        padded_rhs_features=int(padded_rhs),
    )


def resolve_parallelism(ctx: LayerResolveContext) -> None:
    numeric = ctx.numeric()
    if numeric is None:
        raise RuntimeError(f'{ctx.layer_name}: numeric precisions missing before parallelism resolution.')
    tile_cfg = ctx.state['tile_cfg']
    result = _resolve_parallelism_numeric(ctx, numeric, tile_cfg)

    ctx.set_parallelism(result)
    ctx.attributes.parallelism.update(
        {
            'parallel_factor': int(result.parallel_factor),
            'cas_num': int(result.cas_num),
            'cas_length': int(result.cas_length),
            'lhs_tile_bytes': int(result.lhs_tile_bytes),
            'rhs_tile_bytes': int(result.rhs_tile_bytes),
            'output_tile_bytes': int(result.output_tile_bytes),
            'lhs_alignment': int(result.lhs_alignment),
            'rhs_alignment': int(result.rhs_alignment),
        }
    )
    ctx.attributes.slices.update(
        {
            'lhs': int(result.lhs_slice),
            'lhs_raw': int(result.lhs_slice_raw),
            'rhs': int(result.rhs_slice),
            'rhs_raw': int(result.rhs_slice_raw),
        }
    )


def resolve_flags(ctx: LayerResolveContext) -> None:
    fused_trait = ctx.node.traits.get('fused_activation')
    activation = (fused_trait.data.get('activation') if fused_trait else '') or ''
    ctx.attributes.flags['use_relu'] = activation.lower() == 'relu'


def resolve_io_route(ctx: LayerResolveContext) -> None:
    """
    Normalize and default IO routing:
      io_route.inputs.<tensor>  = "direct" | "memtile" | "plio"
      io_route.outputs.<tensor> = "direct" | "memtile" | "plio"

    Meaning:
      - direct:  attempt direct kernel<->kernel transport (only if legal; else memtile fallback in plan pass)
      - memtile: force shared_buffer transport
      - plio:    export/import through extra top_graph ports (intermediate debug IO)
    """
    r = ctx.attributes.io_route
    r.setdefault('inputs', {})
    r.setdefault('outputs', {})

    for t in ctx.node.inputs:
        r['inputs'].setdefault(t.name, 'auto')

    for t in ctx.node.outputs:
        r['outputs'].setdefault(t.name, 'auto')

    user = ctx.node.directives.get('io_route', {})
    for d in ('inputs', 'outputs'):
        if isinstance(user.get(d), dict):
            r[d].update(user[d])


def resolve_io_view(ctx: LayerResolveContext) -> None:
    io_trait = ctx.node.traits.get('io_view')
    if io_trait is None:
        io_trait = TraitInstance('io_view', {'inputs': {}, 'outputs': {}})
        ctx.node.add_trait(io_trait)
    data = io_trait.data
    data.setdefault('inputs', {})
    data.setdefault('outputs', {})

    gen = (getattr(ctx.device, 'generation', '') or '').upper()
    max_rank = 5 if 'AIE-MLV2' in gen else 4

    def _default_view(rank: int) -> Dict[str, Any]:
        return {
            'layout': 'channels_last',
            'independent_axes': list(range(rank - 1)),
            'buffer_order': list(reversed(range(rank))),
        }

    for t in ctx.node.inputs:
        logical = [int(x) for x in t.shape]
        rank = len(logical)
        if rank > max_rank:
            raise ValueError(
                f'{ctx.layer_name}: tensor rank {rank} exceeds max {max_rank} for {ctx.device.generation}.'
            )
        if t.name not in data['inputs']:
            data['inputs'][t.name] = _default_view(rank)

    for t in ctx.node.outputs:
        logical = [int(x) for x in t.shape]
        rank = len(logical)
        if rank > max_rank:
            raise ValueError(
                f'{ctx.layer_name}: tensor rank {rank} exceeds max {max_rank} for {ctx.device.generation}.'
            )
        if t.name not in data['outputs']:
            data['outputs'][t.name] = _default_view(rank)


def _acc_tag_from_width(width: int) -> Optional[str]:
    for tag, bits in ACC_TAG_WIDTHS.items():
        if bits == width:
            return tag
    return None


def _infer_accumulator_tag(
    device: Any,
    input_dtype: Optional[AIEDataType],
    weight_dtype: Optional[AIEDataType],
    acc_precision: Optional[AIEDataType],
) -> Optional[str]:
    if acc_precision is not None and acc_precision.width:
        tag = _acc_tag_from_width(int(acc_precision.width))
        if tag is None:
            raise ValueError(
                f'Unsupported accumulator precision width {acc_precision.width}; ' 'expected one of 32, 48 or 64 bits.'
            )
        return tag

    if input_dtype is None or weight_dtype is None:
        return None

    in_w = int(getattr(input_dtype, 'width', 0) or 0)
    w_w = int(getattr(weight_dtype, 'width', 0) or 0)
    norm_gen = (getattr(device, 'generation', '') or '').upper()
    is_ml = norm_gen.startswith('AIE-ML') or 'XDNA' in norm_gen

    if not is_ml:
        if in_w <= 8 and w_w <= 8:
            return 'acc32'
        if in_w <= 16 and w_w <= 16:
            return 'acc48'
        raise ValueError(
            f'No accumulator tag registered for AIE generation "{device.generation}" with '
            f'input {in_w}-bit and weight {w_w}-bit precisions.'
        )

    if max(in_w, w_w) <= 8:
        return 'acc32'
    if {in_w, w_w} in ({8, 16}, {16, 8}):
        return 'acc32'
    if max(in_w, w_w) <= 16:
        return 'acc64'
    raise ValueError(
        f'No accumulator tag registered for AIE generation "{device.generation}" with '
        f'input {in_w}-bit and weight {w_w}-bit precisions.'
    )


def _extract_rounding(src):
    if hasattr(src, 'rounding_mode'):
        return src.rounding_mode or RoundingMode.TRN
    if hasattr(src, 'rounding'):
        return src.rounding
    return RoundingMode.TRN


def _aie_rounding_token(source) -> str:
    mode = _extract_rounding(source)
    token = ROUNDING_TOKEN_MAP.get(mode)
    if token is None:
        raise ValueError(f'Unsupported rounding mode {mode} for AIE kernel.')
    return token


def resolve_scalars(ctx: LayerResolveContext) -> None:
    scalars = ctx.attributes.scalars
    if ctx.policy.requires_numeric and 'shift' not in scalars:
        raise RuntimeError(f'{ctx.layer_name}: missing resolved numeric scalar "shift"')

    parallelism = ctx.parallelism()
    if parallelism is None:
        raise RuntimeError(f'{ctx.layer_name}: parallelism must be resolved before scalar resolution.')
    scalars['padded_independent_extent'] = int(parallelism.padded_independent_extent)
    scalars['real_independent_extent'] = int(parallelism.independent_extent)
    scalars['padded_lhs_features'] = int(parallelism.padded_lhs_features)
    scalars['padded_rhs_features'] = int(parallelism.padded_rhs_features)
    batch_tensor = _input_tensor_for_roles(ctx, *ROLE_ALIASES['lhs'])
    if batch_tensor is None:
        batch_tensor = next((value for value in ctx.node.inputs if not value.is_parameter), None)
    if batch_tensor is None:
        raise RuntimeError(f'{ctx.layer_name}: missing dynamic input tensor for batch size resolution.')
    scalars['batch_size'] = int(batch_tensor.shape[0])
    scalars['io_shapes'] = _resolve_io_shapes(ctx)


def register_layer_policy(name: str, policy: LayerPolicy) -> None:
    LAYER_RESOLVE_REGISTRY[name] = policy


def get_layer_policy(layer_class: str) -> Optional[LayerPolicy]:
    return LAYER_RESOLVE_REGISTRY.get(layer_class)


LAYER_RESOLVE_REGISTRY: Dict[str, LayerPolicy] = {
    'Input': LayerPolicy(
        namespaces=(),
        resolvers=(),
        requires_numeric=False,
    ),
    'Dense': LayerPolicy(
        namespaces=('numeric', 'tiling', 'parallelism', 'slices', 'flags', 'scalars'),
        requires_numeric=True,
        op_family='dense',
        static_parameter_roles=frozenset({'rhs', 'bias'}),
        resolvers=(
            resolve_numeric,
            resolve_tiling,
            resolve_io_view,
            resolve_parallelism,
            resolve_flags,
            resolve_io_route,
            resolve_scalars,
        ),
    ),
    'Activation': LayerPolicy(
        namespaces=('numeric', 'flags'),
        requires_numeric=False,
        resolvers=(
            resolve_numeric,
            resolve_flags,
        ),
    ),
}


__all__ = [
    'LayerPolicy',
    'LayerResolveContext',
    'NumericBundle',
    'ParallelismResult',
    'get_layer_policy',
    'register_layer_policy',
]
