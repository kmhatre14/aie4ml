# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Resolve per-layer AIE attributes using policy-driven resolver pipelines."""

from __future__ import annotations

import logging

from ..ir import ResolvedAttributes, get_backend_context
from ..op_impls import OpImplSelectionContext, get_op_impl_registry
from .base import AIEPass
from .resolve_registry import LayerResolveContext, get_layer_policy

log = logging.getLogger(__name__)


def _quant_meta(node) -> dict:
    """Collect resolver quant metadata from lowering-provided quant bindings."""
    bindings = node.metadata.get('quant_bindings')
    if not isinstance(bindings, dict):
        raise RuntimeError(f'{node.name}: missing quant_bindings metadata.')

    meta: dict = {}

    output_index = bindings.get('output')
    if output_index is not None:
        if not isinstance(output_index, int) or output_index < 0 or output_index >= len(node.outputs):
            raise RuntimeError(f'{node.name}: invalid quant output binding {output_index}.')
        meta['output_precision'] = node.outputs[output_index].precision

    input_bindings = bindings.get('inputs', {})
    if not isinstance(input_bindings, dict):
        raise RuntimeError(f'{node.name}: invalid quant input bindings metadata.')

    for role, index in input_bindings.items():
        if not isinstance(role, str):
            raise RuntimeError(f'{node.name}: invalid quant input role {role!r}.')
        if not isinstance(index, int) or index < 0 or index >= len(node.inputs):
            raise RuntimeError(f'{node.name}: invalid quant binding for role {role}: {index}.')
        meta[f'{role}_precision'] = node.inputs[index].precision

    return meta


def resolve_aie_attributes(ctx, node) -> ResolvedAttributes:
    """Run the registered resolver pipeline for the given IR node."""

    layer_class = node.metadata.get('layer_class')
    policy = get_layer_policy(layer_class)
    layer_name = node.metadata['source_layer']
    attributes = ResolvedAttributes()

    context = LayerResolveContext(
        backend_ctx=ctx,
        node=node,
        layer_name=layer_name,
        layer_class=layer_class,
        policy=policy,
        quant=_quant_meta(node),
        device=ctx.device,
        attributes=attributes,
    )

    for resolver in policy.resolvers:
        resolver(context)

    if ctx.policies.pack:
        pack_policy = ctx.policies.pack
        attributes.pack.setdefault(
            'policy',
            dict(pack_policy) if isinstance(pack_policy, dict) else pack_policy,
        )
    if ctx.policies.cache:
        cache_policy = ctx.policies.cache
        attributes.pack.setdefault(
            'cache',
            dict(cache_policy) if isinstance(cache_policy, dict) else cache_policy,
        )

    for namespace in policy.namespaces:
        attr_value = getattr(attributes, namespace, None)
        if attr_value is None or (isinstance(attr_value, dict) and not attr_value):
            raise RuntimeError(
                f'{layer_name}: resolver pipeline did not populate required attribute namespace "{namespace}".'
            )

    return attributes


class Resolve(AIEPass):
    """Derive per-layer resolved attributes via the registered resolver pipeline."""

    def __init__(self):
        self.name = 'resolve'
        self._registry = get_op_impl_registry()

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        changed = False
        visited = set()
        for node in ctx.ir.logical:
            if node.metadata['layer_class'] == 'Input':
                continue

            resolved = resolve_aie_attributes(ctx, node)
            selection_ctx = OpImplSelectionContext(
                node=node,
                attributes=resolved,
                device_generation=ctx.device.generation,
                metadata=dict(node.metadata),
            )
            variant = self._registry.select(selection_ctx)
            if variant is None:
                raise RuntimeError(f'{node.name}: no implementation variant satisfies resolved attributes.')

            kernel_cfg = variant.build_config(selection_ctx)
            variant.validate_config(selection_ctx, kernel_cfg)
            inst = ctx.ir.execution.get(node.name)
            if inst is not None:
                same_variant = inst.variant.variant_id == variant.variant_id
                same_cfg = inst.config == kernel_cfg
                if same_variant and same_cfg:
                    visited.add(node.name)
                    continue

            ctx.ir.execution.register(node, variant, kernel_cfg)
            visited.add(node.name)
            changed = True

        if ctx.ir.execution.prune(visited):
            changed = True

        return changed
