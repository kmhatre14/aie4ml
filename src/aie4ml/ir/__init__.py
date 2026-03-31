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
)

__all__ = [
    'AIEBackendContext',
    'AIEPipelineIR',
    'ExecutionIR',
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
