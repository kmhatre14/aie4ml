"""Multi-input demo (companion to tutorial_3.py).

tutorial_3.py builds a SINGLE-input Keras model. This script builds a TWO-input model to exercise
the multi-input PL data path: memory_stream emits one `mm2s` DDR->PLIO mover per graph INPUT
tensor, each sized to its own tensor, plus a single `s2mm` output mover.

Why ONNX and not Keras: any 2-input Keras model must combine its inputs with a merge layer
(Add/Concatenate/...), which the hls4ml->aie frontend does not lower yet. The ONNX frontend does
lower `Add`, so the two-input graph is built directly in ONNX here. Everything downstream
(placement, PL movers, host) is identical to the Keras flow.

Model (two inputs of DIFFERENT shape -> non-byte-identical movers):

    a (int8, [1,64,64])  --DQ--> LayerNorm --Q/DQ--> MatMul(W:[64,128]) --Q/DQ--\
                                                                                  Add --Q/DQ--> y
    b (int8, [1,64,128]) --DQ--> --------------------------------------------- /

  -> input 'a' spans PLIO ports [0,1], input 'b' spans [2,3]:
       mm2s   : N_IFM_STREAMS=2, feeds PLIO_ifm_0,1  (tensor a)
       mm2s_2 : N_IFM_STREAMS=2, feeds PLIO_ifm_2,3  (tensor b, deeper ping-pong: b is larger)
"""

import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

# aie4ml is imported from ../src (run in-place without installing).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from aie4ml.frontends.onnx import from_onnx

PART = 'xilinx_vek280_base_202520_1'
PROJECT_NAME = 'proj_aie_multi_input'
BATCH, ROWS, FEAT_A, FEAT_B = 1, 64, 64, 128
ITERATIONS = 10

INT8_SCALE = float(2.0 ** -4)
LN_EPSILON = float(INT8_SCALE * INT8_SCALE)


# ── tiny int8 quant helpers (QuantizeLinear/DequantizeLinear pairs) ──
def _qparams(prefix):
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [INT8_SCALE]),
        helper.make_tensor(f'{prefix}_zp', TensorProto.INT8, [], [0]),
    ]


def _dq(src, out, prefix):
    return helper.make_node('DequantizeLinear', [src, f'{prefix}_scale', f'{prefix}_zp'], [out], name=f'{prefix}_dq')


def _qdq(nodes, src, out, prefix):
    nodes.append(helper.make_node('QuantizeLinear', [src, f'{prefix}_scale', f'{prefix}_zp'], [f'{prefix}_q'], name=f'{prefix}_q'))
    nodes.append(_dq(f'{prefix}_q', out, prefix))


def build_model():
    a = helper.make_tensor_value_info('a_i8', TensorProto.INT8, [BATCH, ROWS, FEAT_A])
    b = helper.make_tensor_value_info('b_i8', TensorProto.INT8, [BATCH, ROWS, FEAT_B])
    y = helper.make_tensor_value_info('y_dq', TensorProto.FLOAT, [BATCH, ROWS, FEAT_B])

    nodes = [
        _dq('a_i8', 'a_dq', 'a'),
        _dq('b_i8', 'b_dq', 'b'),
        _dq('dense_w_q', 'dense_w_dq', 'dense_w'),
        helper.make_node('LayerNormalization', ['a_dq', 'ln_gamma', 'ln_beta'], ['ln_raw'], name='pre_ln', axis=-1, epsilon=LN_EPSILON),
    ]
    _qdq(nodes, 'ln_raw', 'ln_dq', 'ln_out')
    nodes.append(helper.make_node('MatMul', ['ln_dq', 'dense_w_dq'], ['dense_raw'], name='dense0'))  # [.,64]x[64,128]->[.,128]
    _qdq(nodes, 'dense_raw', 'dense_dq', 'dense_out')
    nodes.append(helper.make_node('Add', ['dense_dq', 'b_dq'], ['res_raw'], name='res_add'))         # + second input b
    _qdq(nodes, 'res_raw', 'y_dq', 'res_out')

    inits = []
    for prefix in ('a', 'b', 'dense_w', 'ln_out', 'dense_out', 'res_out'):
        inits.extend(_qparams(prefix))
    rng = np.random.default_rng(0)
    inits += [
        numpy_helper.from_array(rng.integers(-2, 3, size=(FEAT_A, FEAT_B), dtype=np.int8), name='dense_w_q'),
        numpy_helper.from_array(np.ones((FEAT_A,), dtype=np.float32), name='ln_gamma'),
        numpy_helper.from_array(np.zeros((FEAT_A,), dtype=np.float32), name='ln_beta'),
    ]
    graph = helper.make_graph(nodes, 'aie4ml_multi_input', [a, b], [y], initializer=inits)
    return helper.make_model(graph, producer_name='aie4ml_multi_input', opset_imports=[helper.make_opsetid('', 17)], ir_version=8)


CONFIG = {
    'Part': PART,
    'AIEConfig': {
        'BatchSize': BATCH,
        'Iterations': ITERATIONS,
        'Target': 'hardware',              # emit PL movers + XRT host (not just AIE sim)
        'PLMemory': 'uram',
        'EnablePLTiming': True,            # auto-disabled when >1 input mover (host-side timing instead)
        'PLDataMoverMode': 'memory_stream',  # multi-input requires memory_stream (one mm2s per input tensor)
    },
    'LayerDirectives': {
        'pre_ln': {'parallelism': {'cas_num': 2, 'cas_length': 1}},
        'dense0': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
        'res_add': {'parallelism': {'cas_num': 2, 'cas_length': 1}},
    },
}


if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent / PROJECT_NAME
    aie_model = from_onnx(build_model(), CONFIG, output_dir=out_dir, project_name=PROJECT_NAME)
    aie_model.write()   # generate the project (AIE graph + PL movers + host). Use .build() for full hw.

    # Report the emitted input movers (one mm2s CU per graph input tensor).
    plan = aie_model.context.ir.physical.plan
    print('\n' + '=' * 60)
    print('MULTI-INPUT PL DATA PATH')
    print('=' * 60)
    print(f'graph inputs  : {plan["graph_input_count"]} PLIO port(s)')
    print(f'graph outputs : {plan["graph_output_count"]} PLIO port(s)')
    movers = sorted((out_dir / 'pl').glob('mm2s*.cpp'))
    print(f'input movers  : {len(movers)} mm2s CU(s)')
    import re
    for mv in movers:
        src = mv.read_text()
        streams = re.search(r'N_IFM_STREAMS = (\d+)', src).group(1)
        depth = re.search(r'MAX_IFM_512_PER_STREAM = (\d+)', src).group(1)
        print(f'  {mv.stem:8s}: N_IFM_STREAMS={streams}, ping-pong depth={depth} 512-bit words')
    print(f'\nProject written to: {out_dir}')
    print('Full hardware build:  aie_model.build()   (or `make` in the project dir)')
