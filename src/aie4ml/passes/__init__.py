"""Optimizer passes specific to the AIE backend."""

from .base import AIEPass, run_aie_passes
from .compact_buffer_rank import CompactBufferRank
from .fanout_legalize import LegalizeFanoutEntries
from .fold_apply_alpha import FoldApplyAlpha
from .fold_output_scale import FoldOutputScale
from .fold_views import FoldViewOps
from .force_float_mode import ForceFloatMode
from .fuse_activation import FuseActivationCasts
from .memory_plan import BuildMemoryPlan, CollectMemoryEntries, MaterializeMemoryPlan
from .memtile_legalize import LegalizeMemtilePortLimits
from .pack import PackKernelArtifacts
from .placement import PlaceKernels
from .resolve import Resolve
from .transport_classify import ClassifyTransportEntries

__all__ = [
    'AIEPass',
    'run_aie_passes',
    'FuseActivationCasts',
    'FoldApplyAlpha',
    'FoldOutputScale',
    'ForceFloatMode',
    'FoldViewOps',
    'CompactBufferRank',
    'LegalizeFanoutEntries',
    'LegalizeMemtilePortLimits',
    'Resolve',
    'PackKernelArtifacts',
    'PlaceKernels',
    'CollectMemoryEntries',
    'ClassifyTransportEntries',
    'MaterializeMemoryPlan',
    'BuildMemoryPlan',
]
