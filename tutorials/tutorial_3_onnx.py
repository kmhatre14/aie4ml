import numpy as np
import onnx
from aie4ml import from_onnx
from aie4ml.simulation import read_aie_report
from onnx import TensorProto, helper, numpy_helper

np.random.seed(42)

N_IN = 256
N_OUT = 128
BATCH = 256
ITERS = 4
PLATFORM = 'xilinx_vek280_base_202520_1'

PROJECT_NAME = 'tutorial_3_onnx'

FRAC = 4
SCALE = float(2.0**-FRAC)
ZERO_POINT = 0


def _qparams(prefix):
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [SCALE]),
        helper.make_tensor(f'{prefix}_zp', TensorProto.INT8, [], [ZERO_POINT]),
    ]


def _dequantize(src, out, prefix):
    return helper.make_node('DequantizeLinear', [src, f'{prefix}_scale', f'{prefix}_zp'], [out], name=f'{prefix}_dq')


def build_model():
    x_i8 = helper.make_tensor_value_info('x_i8', TensorProto.INT8, [BATCH, N_IN])
    y_dq = helper.make_tensor_value_info('y_dq', TensorProto.FLOAT, [BATCH, N_OUT])
    nodes = [
        _dequantize('x_i8', 'x_dq', 'x'),
        _dequantize('w_q', 'w_dq', 'w'),
        helper.make_node('MatMul', ['x_dq', 'w_dq'], ['y_raw'], name='dense_0'),
        helper.make_node('QuantizeLinear', ['y_raw', 'y_scale', 'y_zp'], ['y_q'], name='quant_out'),
        _dequantize('y_q', 'y_dq', 'y'),
    ]
    weights = np.random.default_rng(0).integers(-2, 3, size=(N_IN, N_OUT), dtype=np.int8)
    initializers = _qparams('x') + _qparams('w') + _qparams('y')
    initializers.append(numpy_helper.from_array(weights, name='w_q'))
    graph = helper.make_graph(nodes, 'single_dense_large', [x_i8], [y_dq], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)


model = build_model()

# ── ONNX config ──
config = {
    'Part': PLATFORM,
    'AIEConfig': {
        'BatchSize': BATCH,
        'Iterations': ITERS,
        'Target': 'hardware',  # hardware | aie
        'PLMemory': 'uram',  # uram | bram
        'EnablePLTiming': True,  # True | False
        'PLDataMoverMode': 'memory_stream',  # benchmark | memory_stream | external_stream
    },
}

# ── AIE conversion ──
aie_model = from_onnx(
    model,
    config,
    output_dir='proj_aie_' + PROJECT_NAME,
    project_name='proj_aie_' + PROJECT_NAME,
)

aie_model.compile()
aie_model.build()

# Simulation
x = np.random.random((BATCH, N_IN)).astype(np.float32)
y_aie = aie_model.predict(x, simulator='aie')[:BATCH]

report = read_aie_report(aie_model)

print('\n' + '=' * 60)
print('AIE SIMULATION REPORT')
print('=' * 60)

if 'throughput' in report:
    t = report['throughput']
    print('\n[Throughput]')
    print(f"  Avg : {t['Avg_GOPs']} GOPs")
    print(f"  Min : {t['Min_GOPs']} GOPs")
    print(f"  Max : {t['Max_GOPs']} GOPs")

if 'output_interval' in report:
    ii = report['output_interval']
    print('\n[Output Interval (ns)]')
    for name, vals in ii.items():
        if isinstance(vals, dict):
            print(f'  {name}:')
            for k, v in vals.items():
                print(f'    {k}: {v} ns')
