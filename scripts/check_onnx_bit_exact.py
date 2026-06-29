#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from aie4ml.frontends.onnx import from_onnx  # noqa: E402
from aie4ml.simulation import build_io_layout  # noqa: E402


def _specialize_dynamic_batch(proto: onnx.ModelProto, batch: int) -> onnx.ModelProto:
    for value_info in list(proto.graph.input) + list(proto.graph.output) + list(proto.graph.value_info):
        shape = value_info.type.tensor_type.shape
        for axis, dim in enumerate(shape.dim):
            is_dynamic = dim.dim_param or not dim.HasField('dim_value')
            if not is_dynamic:
                continue
            if axis != 0:
                raise ValueError(
                    f'{value_info.name}: only dynamic batch dimension is supported by this checker; '
                    f'axis {axis} is dynamic.'
                )
            dim.ClearField('dim_param')
            dim.dim_value = int(batch)
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


def _load_config(path: Path | None, *, part: str, batch: int) -> dict:
    config = {} if path is None else json.loads(path.read_text())
    config['Part'] = part
    aie_config = dict(config.get('AIEConfig', {}))
    aie_config.update({'BatchSize': int(batch), 'Iterations': 1})
    config['AIEConfig'] = aie_config
    return config


def _input_dtype(type_string: str):
    try:
        return {
            'tensor(float)': np.float32,
            'tensor(int8)': np.int8,
            'tensor(uint8)': np.uint8,
            'tensor(int16)': np.int16,
            'tensor(uint16)': np.uint16,
        }[type_string]
    except KeyError as exc:
        raise ValueError(f'unsupported ONNX input type {type_string!r}') from exc


def _input_feeds(session: ort.InferenceSession, *, batch: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds = {}
    for input_info in session.get_inputs():
        shape = [batch if dim is None or isinstance(dim, str) else int(dim) for dim in input_info.shape]
        dtype = _input_dtype(input_info.type)
        if np.issubdtype(dtype, np.floating):
            value = (rng.random(shape, dtype=np.float32) * 2.0) - 1.0
        else:
            info = np.iinfo(dtype)
            lo = max(info.min, -96)
            hi = min(info.max, 96)
            value = rng.integers(lo, hi + 1, size=shape, dtype=dtype)
        feeds[input_info.name] = np.asarray(value, dtype=dtype)
    return feeds


def _quantize_inputs(session: ort.InferenceSession, requested: bool | None) -> bool:
    if requested is not None:
        return bool(requested)
    float_inputs = [item.name for item in session.get_inputs() if np.issubdtype(_input_dtype(item.type), np.floating)]
    if not float_inputs:
        return False
    if len(float_inputs) != len(session.get_inputs()):
        raise ValueError('mixed float/integer graph inputs require explicit --quantize-in or --no-quantize-in.')
    return True


def _normalize_aie_outputs(aie_outputs, output_names: list[str]) -> list[np.ndarray]:
    if isinstance(aie_outputs, dict):
        missing = [name for name in output_names if name not in aie_outputs]
        extra = [name for name in aie_outputs if name not in output_names]
        if missing or extra:
            raise RuntimeError(
                f'Output layout mismatch: missing={missing} extra={extra} '
                f'plan={output_names} AIE={list(aie_outputs)}'
            )
        return [np.asarray(aie_outputs[name]) for name in output_names]
    if isinstance(aie_outputs, (list, tuple)):
        return [np.asarray(output) for output in aie_outputs]
    return [np.asarray(aie_outputs)]


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare ONNX Runtime outputs against raw AIE x86sim outputs.')
    parser.add_argument('model', type=Path, help='Path to the ONNX model')
    parser.add_argument('--part', default='xilinx_vek280_base_202520_1')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--config', type=Path, help='JSON compiler config, including optional LayerDirectives')
    parser.add_argument(
        '--quantize-in',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override input quantization; default infers true for float inputs and false for integer inputs',
    )
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
    config = _load_config(args.config, part=args.part, batch=args.batch)

    proto = _specialize_dynamic_batch(onnx.load(str(args.model)), args.batch)

    aie_model = from_onnx(proto, config, output_dir=output_dir)
    aie_model.write()
    aie_model.compile()

    session = ort.InferenceSession(proto.SerializeToString(), providers=['CPUExecutionProvider'])
    feeds = _input_feeds(session, batch=args.batch, seed=args.seed)
    reference_outputs = [np.asarray(output) for output in session.run(None, feeds)]
    aie_outputs = aie_model.predict(
        feeds,
        simulator='x86',
        quantize_in=_quantize_inputs(session, args.quantize_in),
        dequantize_out=False,
    )

    layout = build_io_layout(aie_model)
    output_names = list(layout.outputs.keys())
    aie_outputs = _normalize_aie_outputs(aie_outputs, output_names)

    if len(output_names) != len(aie_outputs):
        raise RuntimeError(f'Output layout mismatch: plan={len(output_names)} AIE={len(aie_outputs)}')
    if len(reference_outputs) != len(aie_outputs):
        raise RuntimeError(f'ONNX output count mismatch: reference={len(reference_outputs)} AIE={len(aie_outputs)}')

    failed = False
    for index, (name, ref, aie) in enumerate(zip(output_names, reference_outputs, aie_outputs)):
        dtype = layout.outputs[name][0].dtype
        ref_q = _requantize_output(ref, width=dtype.width, signed=dtype.signed, frac=dtype.frac)
        if ref_q.shape != aie.shape:
            raise RuntimeError(f'output[{index}] shape mismatch: {ref_q.shape} vs AIE {aie.shape}')
        diff = np.abs(ref_q.astype(np.int64, copy=False) - aie.astype(np.int64, copy=False))
        max_diff = int(diff.max()) if diff.size else 0
        mean_diff = float(diff.mean()) if diff.size else 0.0
        within_atol = bool(max_diff <= args.atol)
        status = 'bit_exact' if max_diff == 0 else (f'within_atol={args.atol}' if within_atol else 'MISMATCH')
        print(f'output[{index}] {name}  max|dq|={max_diff}  mean|dq|={mean_diff:.6g}  {status}')
        failed |= not within_atol
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
