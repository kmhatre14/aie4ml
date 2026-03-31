#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from aie4ml.frontends.onnx import from_onnx  # noqa: E402
from aie4ml.simulation import build_io_layout  # noqa: E402


def _drop_scalar_outputs(proto: onnx.ModelProto) -> onnx.ModelProto:
    semantic = [output for output in proto.graph.output if output.type.tensor_type.shape.dim]
    if len(semantic) != len(proto.graph.output):
        del proto.graph.output[:]
        proto.graph.output.extend(semantic)
    return proto


def _requantize_output(values: np.ndarray, *, width: int, signed: bool, frac: int) -> np.ndarray:
    scale = float(1 << int(frac))
    quantized = np.rint(np.asarray(values, dtype=np.float64) * scale).astype(np.int64, copy=False)
    if signed:
        lo = -(1 << (int(width) - 1))
        hi = (1 << (int(width) - 1)) - 1
    else:
        lo = 0
        hi = (1 << int(width)) - 1
    return np.clip(quantized, lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare ONNX outputs against raw AIE x86sim outputs.')
    parser.add_argument('model', type=Path, help='Path to the ONNX model')
    parser.add_argument('--part', default='xilinx_vek280_base_202520_1')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument(
        '--atol',
        type=int,
        default=1,
        help='Accepted integer tolerance (default 1). '
        'A tolerance of 1 accounts for the 1-LSB rounding difference between '
        'the float32 onnxruntime reference path and the exact integer '
        'accumulation in the AIE kernel. Use --atol 0 for strict comparison.',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (REPO_ROOT / 'scripts' / f'bitexact_{args.model.stem}')
    config = {
        'Part': args.part,
        'AIEConfig': {'BatchSize': args.batch, 'Iterations': 1},
    }

    proto = _drop_scalar_outputs(onnx.load(str(args.model)))

    aie_model = from_onnx(proto, config, output_dir=output_dir)
    aie_model.write()
    aie_model.compile()

    session = ort.InferenceSession(proto.SerializeToString(), providers=['CPUExecutionProvider'])
    input_info = session.get_inputs()[0]
    input_shape = [args.batch if dim is None or dim == 'batch' else dim for dim in input_info.shape]

    rng = np.random.default_rng(args.seed)
    x = (rng.random(input_shape, dtype=np.float32) * 2.0) - 1.0

    ref_outputs = [np.asarray(output) for output in session.run(None, {input_info.name: x})]
    aie_outputs = aie_model.predict(x, simulator='x86', quantize_in=True, dequantize_out=False)
    if not isinstance(aie_outputs, (list, tuple)):
        aie_outputs = [aie_outputs]
    aie_outputs = [np.asarray(output) for output in aie_outputs]

    if len(ref_outputs) != len(aie_outputs):
        raise RuntimeError(f'Output count mismatch: ONNX={len(ref_outputs)} AIE={len(aie_outputs)}')

    layout = build_io_layout(aie_model)
    output_names = list(layout.outputs.keys())
    if len(output_names) != len(ref_outputs):
        raise RuntimeError(f'Output layout mismatch: plan={len(output_names)} ONNX={len(ref_outputs)}')

    for index, (name, ref, aie) in enumerate(zip(output_names, ref_outputs, aie_outputs)):
        dtype = layout.outputs[name][0].dtype
        ref_q = _requantize_output(ref, width=dtype.width, signed=dtype.signed, frac=dtype.frac)
        if ref_q.shape != aie.shape:
            raise RuntimeError(f'output[{index}] shape mismatch: ONNX {ref_q.shape} vs AIE {aie.shape}')
        diff = np.abs(ref_q.astype(np.int64, copy=False) - aie.astype(np.int64, copy=False))
        max_diff = int(diff.max()) if diff.size else 0
        mean_diff = float(diff.mean()) if diff.size else 0.0
        within_atol = bool(max_diff <= args.atol)
        status = 'bit_exact' if max_diff == 0 else (f'within_atol={args.atol}' if within_atol else 'MISMATCH')
        note = '  (float32 ref rounding artifact)' if max_diff > 0 and within_atol else ''
        print(f'output[{index}] {name}  max|dq|={max_diff}  mean|dq|={mean_diff:.6g}  {status}{note}')
        if not within_atol:
            sys.exit(1)


if __name__ == '__main__':
    main()
