from __future__ import annotations

from ..ir import TraitDefinition


def register_default_traits(ctx) -> None:
    ctx.traits.register(
        TraitDefinition(
            name='fused_activation',
            dialects=(ctx.device.dialect,),
            fields=('activation',),
            description='Indicates that an activation has been fused into the producer op.',
        )
    )
    ctx.traits.register(
        TraitDefinition(
            name='io_view',
            dialects=(ctx.device.dialect,),
            fields=('inputs', 'outputs'),
            description='Per-tensor logical-to-physical view mapping for IO/staging.',
        )
    )


def attach_quant_role_bindings(meta: dict) -> dict:
    roles = list(meta.get('input_roles') or [])
    meta['quant_bindings'] = {
        'inputs': {role: index for index, role in enumerate(roles)},
        'output': 0,
    }
    return meta
