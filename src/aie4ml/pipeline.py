"""Core AIE compiler pipeline definition."""

from .passes import (
    CollectMemoryEntries,
    CompactBufferRank,
    FoldApplyAlpha,
    FoldTransposeViews,
    ForceFloatMode,
    FuseActivationCasts,
    LegalizeFanoutEntries,
    LegalizeMemtilePortLimits,
    MaterializeMemoryPlan,
    PackKernelArtifacts,
    PlaceKernels,
    Resolve,
)

HLS4ML_FLOW_SPEC = (
    ('force_float', ForceFloatMode),
    ('fold_apply_alpha', FoldApplyAlpha),
    ('fuse', FuseActivationCasts),
    ('fold_views', FoldTransposeViews),
    ('resolve', Resolve),
    ('pack', PackKernelArtifacts),
    ('placement', PlaceKernels),
    ('memory_collect', CollectMemoryEntries),
    ('fanout_legalize', LegalizeFanoutEntries),
    ('memtile_legalize', LegalizeMemtilePortLimits),
    ('memory_plan', MaterializeMemoryPlan),
    ('compact_batch', CompactBufferRank),
)

DEFAULT_PIPELINE = tuple(pass_cls for _, pass_cls in HLS4ML_FLOW_SPEC)
