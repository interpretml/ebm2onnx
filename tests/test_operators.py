import ebm2onnx.graph as graph
import ebm2onnx.operators as ops

import numpy as np
import onnx

from .utils import assert_model_result


def test_add():
    g = graph.create_graph()

    a = graph.create_initializer(g, "a", onnx.TensorProto.FLOAT, [1], [0.3])
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None])

    l = ops.add()(graph.merge(i, a))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None])
    
    assert_model_result(l, 
        input={
            'i': [0.1, 1.2, 11, 4.2],
        },
        expected_result=[[0.4, 1.5, 11.3, 4.5]]
    )


def test_cast():
    g = graph.create_graph()

    i = graph.create_input(g, "i", onnx.TensorProto.INT64, [None, 1])

    l = ops.cast(onnx.TensorProto.FLOAT)(i)
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None, 1])
    
    assert_model_result(l, 
        input={
            'i': [
                [1],
                [2],
                [11],
                [4],
            ]
        },
        expected_result=[[
            [1.0],
            [2.0],
            [11.0],
            [4.0]
        ]]
    )


def test_flatten():
    g = graph.create_graph()

    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None])

    l = ops.flatten()(i)
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None, 1])
    
    assert_model_result(l, 
        input={
            'i': [0.1, 0.2, 0.3, 0.4]
        },
        expected_result=[[
            [0.1],
            [0.2],
            [0.3],
            [0.4]
        ]]
    )


def test_less():
    g = graph.create_graph()

    a = graph.create_initializer(g, "a", onnx.TensorProto.FLOAT, [4], [1.1, 2.3, 3.5, 9.6])
    b = graph.create_input(g, "b", onnx.TensorProto.FLOAT, [None, 1])

    l = ops.less()(graph.merge(a, b))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.BOOL, [None, 4])
    
    assert_model_result(l, 
        input={
            'b': [
                [0.1],
                [1.2],
                [11],
                [4.2],
                [np.NaN],
            ]
        },
        expected_result=[[
            [False, False, False, False],
            [True, False, False, False],
            [True, True, True, True],
            [True, True, True, False],
            [False, False, False, False],
        ]]
    )


def test_mul():
    g = graph.create_graph()

    a = graph.create_initializer(g, "a", onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])
    b = graph.create_input(g, "b", onnx.TensorProto.FLOAT, [None, 3])

    l = ops.mul()(graph.merge(a, b))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None, 3])
    
    assert_model_result(l, 
        input={
            'b': [
                [0.1, 0.1, 0.1],
                [0.1, 0.2, 0.3],
            ]
        },
        expected_result=[[
            [0.1, 0.2, 0.3],
            [0.1, 0.4, 0.9],
        ]]
    )


def test_argmax():
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 3])

    l = ops.argmax(axis=1)(i)
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.INT64, [None, 1])
    
    assert_model_result(l, 
        input={
            'i': [
                [1, 4, 2],
                [2, 8, 12],
                [11, 0, 5],
            ]
        },
        expected_result=[[
            [1],
            [2],
            [0],
        ]]
    )


def test_gather_elements():
    g = graph.create_graph()

    a = graph.create_initializer(g, "a", onnx.TensorProto.FLOAT, [3, 1], [0.1, 0.2, 0.3])
    b = graph.create_input(g, "b", onnx.TensorProto.INT64, [None, 1])

    l = ops.gather_elements()(graph.merge(a, b))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None, 1])
    
    assert_model_result(l, 
        input={
            'b': [
                [2],
                [1],
                [0],
            ]
        },
        expected_result=[[
            [0.3],
            [0.2],
            [0.1],
        ]]
    )


def test_gather_nd():
    g = graph.create_graph()

    a = graph.create_initializer(g, "a", onnx.TensorProto.FLOAT, [3, 3], np.array([
        [0.1, 0.2, 0.3],
        [1.1, 2.2, 3.3],
        [0.1, 20.2, 30.3],
    ]).flatten())
    b = graph.create_input(g, "b", onnx.TensorProto.INT64, [None, 2])

    l = ops.gather_nd()(graph.merge(a, b))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None])
    
    assert_model_result(l, 
        input={
            'b': [
                [2, 0],
                [1, 1],
                [0, 1],
            ]
        },
        expected_result=np.array([[0.1, 2.2, 0.2]])
    )


def test_concat():
    g = graph.create_graph()

    a = graph.create_input(g, "a", onnx.TensorProto.FLOAT, [3, 1])
    b = graph.create_input(g, "b", onnx.TensorProto.FLOAT, [3, 1])

    l = ops.concat(axis=1)(graph.merge(a, b))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None, 2])
    
    assert_model_result(l, 
        input={
            'a': [[0.1], [0.2], [0.3]],
            'b': [[1.1], [1.2], [1.3]],
        },
        expected_result=[[
            [0.1, 1.1],
            [0.2, 1.2],
            [0.3, 1.3],
        ]]
    )


def test_expand():
    g = graph.create_graph()

    shape = graph.create_initializer(g, "shape", onnx.TensorProto.INT64, [2], [4, 3])
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 1])

    l = ops.expand()(graph.merge(i, shape))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None, 3])

    assert_model_result(l,
        input={
            'i': [
                [0.1],
                [1.2],
                [11],
                [4.2],
            ]
        },
        expected_result=[[
            [0.1, 0.1, 0.1],
            [1.2, 1.2, 1.2],
            [11, 11, 11],
            [4.2, 4.2, 4.2]
        ]],
    )



def test_reduce_sum():
    g = graph.create_graph()

    axis = graph.create_initializer(g, "axis", onnx.TensorProto.INT64, [1], [1])
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 3])

    l = ops.reduce_sum(keepdims=0)(graph.merge(i, axis))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None])
    
    assert_model_result(l, 
        input={
            'i': [
                [0.1, 1.0, 1.2],
                [1.2, 0.4, 0.9],
                [11, 0.8, -0.2],
                [4.2, 3.2, -6.4],
            ]
        },
        expected_result=[[2.3, 2.5, 11.6, 1.0]]
    )


def test_reshape():
    g = graph.create_graph()

    shape = graph.create_initializer(g, "shape", onnx.TensorProto.INT64, [1], [0])
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 1])

    l = ops.reshape()(graph.merge(i, shape))
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None])
    
    assert_model_result(l, 
        input={
            'i': [
                [0.1],
                [1.2],
                [11],
                [4.2],
            ]
        },
        expected_result=[[0.1, 1.2, 11, 4.2]]
    )


def test_softmax():
    g = graph.create_graph()

    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 2])

    l = ops.softmax()(i)
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.FLOAT, [None, 2])
    
    assert_model_result(l, 
        input={
            'i': [
                [0.0, 0.68],
                [0.0, 0.2],
                [1.2, 0.3],
                [0.0, -0.2],
            ]
        },
        expected_result=[[
            [0.3362613 , 0.66373867],
            [0.450166  , 0.54983395],
            [0.7109495 , 0.2890505 ],
            [0.54983395, 0.450166  ]
        ]],
    )
