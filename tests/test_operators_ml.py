import ebm2onnx.graph as graph
import ebm2onnx.operators_ml as mlops

import numpy as np
import onnx

from .utils import assert_model_result, infer_model


def test_category_mapper_str2int():
    g = graph.create_graph()

    i = graph.create_input(g, "i", onnx.TensorProto.STRING, [None])

    l = mlops.category_mapper(
        cats_int64s=[0, 1, 2],
        cats_strings=["foo", "bar", "biz"],
    )(i)
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.INT64, [None])
    
    result = infer_model(graph.compile(l, target_opset=13),
        input={
            'i': ["biz", "foo", "bar", "flah"],
        }
    )
    expected_result = [2, 0, 1, -1]
    assert len(expected_result) == len(result[0])
    for i, r in enumerate(expected_result):
        assert result[0][i] == r


def test_category_mapper_int2str():
    g = graph.create_graph()

    i = graph.create_input(g, "i", onnx.TensorProto.INT64, [None])

    l = mlops.category_mapper(
        cats_int64s=[0, 1, 2],
        cats_strings=["foo", "bar", "biz"],
    )(i)
    l = graph.add_output(l, l.transients[0].name, onnx.TensorProto.STRING, [None])
    
    result = infer_model(graph.compile(l, target_opset=13),
        input={
            'i': [2, 0, 1, 8],
        }
    )
    expected_result = ["biz", "foo", "bar", "_Unused"]
    assert len(expected_result) == len(result[0])
    for i, r in enumerate(expected_result):
        assert result[0][i] == r
