"""Public exports for the AIE intermediate representation."""

from .context import (
    AIEBackendContext,
    BackendPolicies,
    DeviceSpec,
    TraitDefinition,
    TraitRegistry,
    ensure_backend_context,
    get_backend_context,
)
from .graph import (
    AIEPipelineIR,
    ExecutionIR,
    LogicalIR,
    OpImplInstance,
    OpNode,
    PhysicalIR,
    ResolvedAttributes,
    TensorVar,
    TraitInstance,
    input_role,
    input_role_map,
    input_tensor_for_role,
)

__all__ = [
    'AIEBackendContext',
    'AIEPipelineIR',
    'ExecutionIR',
    'input_role',
    'input_role_map',
    'input_tensor_for_role',
    'LogicalIR',
    'PhysicalIR',
    'OpImplInstance',
    'OpNode',
    'TensorVar',
    'BackendPolicies',
    'DeviceSpec',
    'ResolvedAttributes',
    'TraitDefinition',
    'TraitInstance',
    'TraitRegistry',
    'ensure_backend_context',
    'get_backend_context',
]
