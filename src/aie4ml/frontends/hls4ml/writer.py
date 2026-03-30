# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""hls4ml Writer adapter for the AIE backend."""

from hls4ml.writer.writers import Writer

from ...ir import get_backend_context
from ...writer import AIEProjectEmitter


class AIEWriter(Writer):
    def __init__(self):
        super().__init__()
        self._emitter = AIEProjectEmitter()

    def write_aie(self, model):
        ctx = get_backend_context(model)
        self._emitter.emit(ctx)
