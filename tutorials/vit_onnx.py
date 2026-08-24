import json
from pathlib import Path
import sys
import onnx
from aie4ml import from_onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tutorials'))

OUT_DIR = Path('tutorials/onnx_models')
OUT_DIR.mkdir(exist_ok=True)
PROJECT_NAME = 'vit_onnx'
ONNX_MODEL = OUT_DIR / 'vit_onnx.onnx'
CONFIG_FILE = OUT_DIR / 'vit_onnx.json'
PLATFORM = 'xilinx_vek280_base_202520_1'
BATCH, SEQ, DIM, HID = 1, 256, 192, 768
HEADS = 3
HEAD_DIM = DIM // HEADS
NUM_CLASSES = 10
PLIO_ALIGN = 16 # 128-bit PLIO word = 16 values (10 + 6)
PADDED_CLASSES = -(-NUM_CLASSES // PLIO_ALIGN) * PLIO_ALIGN
INV_SQRT = 0.125 # Scaling factor: 1/8
EPS = float(2.0 ** -10) # For layerNorm
FRAC = { # Number of fractional bits per tensor 
    # value = integerVal * 2^(-fracBits)
    'x':5, 
    'wq':7, 'wk':7, 'wv':7, 'wo':7, 'w1':7, 'w2':7, 'wh': 7,
    'bq':5, 'bk':5, 'bv':5, 'bo':5, 'b1':5, 'b2':5, 'bh':5,
    'ln1':5, 'ln2':5, 'hln': 5,
    'q':5, 'k':5, 'v':5,
    'sc':3, # Attn scores: 8 - 4 - 1 = 3, quantized_relu(8,0)
    'attn':8,  # quantized_relu(8, 0)
    'ctx':5, 'proj':5, 'res1':5, 
    'h1':5, 'h1r':7, 'h2':5, 'res2':5,
    'head': 3
}
weights = {
    # w(q,k,v,o): (192, 192), w1: (192,768), w2: (192, 16)
    'wq': (DIM, DIM), 'wk': (DIM, DIM), 'wv': (DIM, DIM), 'wo': (DIM, DIM),
    'w1': (DIM, HID), 'w2': (HID, DIM),
    'wh': (DIM, PADDED_CLASSES)
}
biases = {
    'bq': DIM, 'bk': DIM, 'bv': DIM, 'bo': DIM,
    'b1': HID, 'b2': DIM, 'bh': PADDED_CLASSES,
}

# ONNX scale + zero point tensors for symmetric quantization
def _qparams(prefix):
    s = float(2.0 ** -FRAC[prefix])
    return [helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [s]),
            helper.make_tensor(f'{prefix}_zp', TensorProto.INT8, [], [0])]

def _dequantize(src, out, prefix):
    return helper.make_node('DequantizeLinear', [src, f'{prefix}_scale', f'{prefix}_zp'], [out], name=f'{prefix}_dq')

def qdq(nodes, raw, prefix, out=None):
    q = f'{prefix}_q'
    dq = out or f'{prefix}_dq'
    nodes.extend([
        helper.make_node('QuantizeLinear', [raw, f'{prefix}_scale', f'{prefix}_zp'], [q], name=q),
        _dequantize(q, dq, prefix)
    ])
    return dq

# Normal ONNX op + qdq
def op(nodes, opType, inputs, raw, prefix, name, out=None, **attrs):
    nodes.append(helper.make_node(opType, inputs, [raw], name=name, **attrs))
    return qdq(nodes, raw, prefix, out)

# Bias (Add) + QDQ
def biased_op(nodes, opType, inputs, raw, prefix, name, bias, out=None, **attrs):
    nodes.append(helper.make_node(opType, inputs, [raw], name=name, **attrs))
    biased = f'{raw}_biased'
    nodes.append(helper.make_node('Add', [raw, f'{bias}_dq'], [biased], name=f'{name}_bias'))
    return qdq(nodes, biased, prefix, out)

def build_model():
    # Input + Output definitions
    # graph input (1, 256, 192)
    x_i8 = helper.make_tensor_value_info('x_i8', TensorProto.INT8, [BATCH, SEQ, DIM])
    # graph output (1, 256, 16)
    y_dq = helper.make_tensor_value_info('y_dq', TensorProto.FLOAT, [BATCH, SEQ, PADDED_CLASSES])
    nodes = []
    nodes.append(_dequantize('x_i8', 'x_dq', 'x')) # input dequant, x_float = x_int8 * 2^-5
    # weight dequant
    for w in weights:
        nodes.append(_dequantize(f'{w}_q', f'{w}_dq', w))
    # bias dequant
    for b in biases:
        nodes.append(_dequantize(f'{b}_q', f'{b}_dq', b))
    #--------------------------------------------------------------------------------------------------------------
    # Attention
    #--------------------------------------------------------------------------------------------------------------
    # (1,256, 192) -> (1,256,192)
    ln1_dq = op(nodes, 'LayerNormalization', ['x_dq', 'g1', 'b1'], 'ln_raw', 'ln1', 'ln1', axis=-1, epsilon=EPS)
    ## Q, K, V projections (1, 256, 192)
    for proj, w in [('q', 'wq'), ('k', 'wk'), ('v', 'wv')]:
        biased_op(nodes, 'MatMul', [ln1_dq, f'{w}_dq'], f'{proj}_raw', proj, f'mm_{proj}', f'b{proj}')
    ## Split the projections into independent attention heads.
    for proj in ('q', 'k', 'v'): # (1,256,192) -> (1,256,3,64)
        nodes.append(helper.make_node('Reshape', [f'{proj}_dq', 'head_shape'],
                                      [f'{proj}_reshaped'], name=f'reshape_{proj}'))
        # (1,3,256,64)
        nodes.append(helper.make_node('Transpose', [f'{proj}_reshaped'], [f'{proj}_heads'],
                                      name=f'transpose_{proj}', perm=[0,2,1,3]))
    ## k transpose for Q @ K^T, preserving the head dimension. (1,3,64,256)
    nodes.append(helper.make_node('Transpose', ['k_heads'], ['kt'], name='transpose_kt', perm=[0,1,3,2]))
    ## Q @ K^T, then scale before score quantization (Keras SCQ ordering).
    # Q: (1, 3, 256, 64), KT: (1, 3, 64, 256), sc_raw: (1, 3, 256, 256)
    nodes.append(helper.make_node('MatMul', ['q_heads', 'kt'], ['sc_raw'], name='mm_qk'))
    # Scaling: 1 / sqrt(64) = 1/8
    nodes.append(helper.make_node('Mul', ['sc_raw', 'inv_sqrt_d'], ['scaled_raw'], name='scale'))
    scaled_dq = qdq(nodes, 'scaled_raw', 'sc', out='scaled_dq')
    ## Softmax (1, 3, 256, 256)
    attn_dq = op(nodes, 'Softmax', [scaled_dq], 'attn_raw', 'attn', 'softmax', axis=-1)
    ## Attention @ V: (1, 3, 256, 256) -> (1, 3, 256, 64)
    ctx_dq = op(nodes, 'MatMul', [attn_dq, 'v_heads'], 'ctx_raw', 'ctx', 'mm_av')
    # Head Merging: (1, 3, 256, 64) -> (1, 256, 3, 64)
    nodes.append(helper.make_node('Transpose', ['ctx_dq'], ['ctx_merged'],
                                  name='transpose_ctx', perm=[0,2,1,3]))
    # Flattens: -> (1, 256, 192)
    nodes.append(helper.make_node('Reshape', ['ctx_merged', 'merge_shape'],
                                  ['ctx_flat'], name='reshape_ctx'))
    ## Output projection (ctx*W_0 + b_0)
    proj_dq = biased_op(nodes, 'MatMul', ['ctx_flat', 'wo_dq'], 'proj_raw', 'proj', 'mm_o', 'bo')
    ## First residual (original input to output)
    res1_dq = op(nodes, 'Add', ['x_dq', proj_dq], 'res1_raw', 'res1', 'res1')
    #----------------------------------------------------------------------------------------------------------------------------
    # MLP
    #----------------------------------------------------------------------------------------------------------------------------
    ## LayerNom
    ln2_dq = op(nodes, 'LayerNormalization', [res1_dq, 'g2', 'b2'], 'ln2_raw', 'ln2', 'ln2', axis=-1, epsilon=EPS)
    ## First projection
    # (1, 256, 192) -> (1, 256, 768)
    h1_dq = biased_op(nodes, 'MatMul', [ln2_dq, 'w1_dq'], 'h1_raw', 'h1', 'mm_1', 'b1')
    ## ReLU
    h1r_dq = op(nodes, 'Relu', [h1_dq], 'h1r_raw', 'h1r', 'relu')
    ## Second projection -> (1, 256, 192)
    h2_dq = biased_op(nodes, 'MatMul', [h1r_dq, 'w2_dq'], 'h2_raw', 'h2', 'mm_2', 'b2')
    res2_dq = op(nodes, 'Add', [res1_dq, h2_dq], 'res2_raw', 'res2', 'res_2')
    #----------------------------------------------------------------------------------------------------------------------------
    # Classification Head
    #----------------------------------------------------------------------------------------------------------------------------
    hln_dq = op(nodes, 'LayerNormalization', [res2_dq, 'gh', 'bh'],'hln_raw', 'hln', 'head_ln', axis=-1, epsilon=EPS)
    # -> (1, 256, 16) (logits = ln(x)W_h + b_h)
    biased_op(nodes, 'MatMul', [hln_dq, 'wh_dq'], 'head_raw', 'head', 'head', 'bh', out='y_dq') # (b, seq, padded)

    initializers = []
    for prefix in FRAC:
        initializers += _qparams(prefix)
    __range = np.random.default_rng(0);
    initializers += [numpy_helper.from_array(__range.integers(-2,3,size=shape,dtype=np.int8), name=f'{name}_q')
                     for name, shape in weights.items()]
    initializers += [numpy_helper.from_array(__range.integers(-2,3,size=size,dtype=np.int8), name=f'{name}_q')
                     for name, size in biases.items()]
    # LayerNorm (gamma=1, beta=0)
    for name in ['g1', 'b1', 'g2', 'b2', 'gh', 'bh']:
        arr = np.ones(DIM, dtype=np.float32) if name[0] == 'g' else np.zeros(DIM, dtype=np.float32)
        initializers.append(numpy_helper.from_array(arr, name=name))
    initializers.append(helper.make_tensor('inv_sqrt_d', TensorProto.FLOAT, [], [INV_SQRT]))
    initializers.append(numpy_helper.from_array(
        np.asarray([BATCH, SEQ, HEADS, HEAD_DIM], dtype=np.int64), name='head_shape'))
    initializers.append(numpy_helper.from_array(
        np.asarray([BATCH, SEQ, DIM], dtype=np.int64), name='merge_shape'))
    graph = helper.make_graph(nodes, 'vit_block', [x_i8], [y_dq], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)


config = {
    'Part': PLATFORM,
    'AIEConfig': {
        'BatchSize': BATCH,
        'Iterations': 1,
        'Target': 'hardware',  # hardware | aie
        'PLMemory': 'uram',  # uram | bram
        'EnablePLTiming': True,  # True | False
        'PLDataMoverMode': 'memory_stream',  # benchmark | memory_stream | external_stream
    }
    # cols = 256 -> B <= 32767/256 ~ 127 && B - S*Dmax = 27
    # 'LayerDirectives': {
    #     'softmax': {
    #         'hccs': {
    #             'B': 127,
    #             'S': 1,
    #             'Dmax': 100
    #         }
    #     },
    # }
}


def main():
    model = build_model()
    onnx.checker.check_model(model)
    onnx.save(model, ONNX_MODEL)
    CONFIG_FILE.write_text(json.dumps(config, indent=4))
    print(f'ONNX model saved to {ONNX_MODEL}')
    print(f'Config file saved to {CONFIG_FILE}')
    aie_model = from_onnx(
        str(ONNX_MODEL),
        config,
        output_dir='proj_aie_' + PROJECT_NAME,
        project_name='proj_aie_' + PROJECT_NAME,
    )
    aie_model.compile()
    aie_model.build()

if __name__ == '__main__':
    main()