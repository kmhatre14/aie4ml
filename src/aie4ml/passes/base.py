# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Base class for AIE passes."""

from __future__ import annotations

from typing import Iterable

from ..ir.context import AIEBackendContext


class AIEPass:
    """Base class for all AIE compiler passes.

    Subclasses set `self.name` and implement `transform(model_or_ctx)`.
    When driven by the hls4ml optimizer, `model_or_ctx` is the hls4ml model.
    When driven by `run_aie_passes`, `model_or_ctx` is a bare `AIEBackendContext`.
    In both cases, call `get_backend_context(model_or_ctx)` to obtain the context.
    Returns True if the IR was modified, False otherwise.
    """

    name: str

    def transform(self, model_or_ctx) -> bool:
        raise NotImplementedError


def run_aie_passes(ctx: AIEBackendContext, passes: Iterable[AIEPass]) -> None:
    """Run a sequence of AIE passes to a fixed point."""
    changed = True
    while changed:
        changed = False
        for p in passes:
            if p.transform(ctx):
                changed = True
