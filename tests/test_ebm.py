import ebm2onnx.ebm as ebm
import ebm2onnx.graph as graph
import ebm2onnx.operators as ops
import onnx
import numpy as np

from .utils import assert_model_result, infer_model


def test_get_bin_index_on_continuous_value():
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 1])

    g = ebm.get_bin_index_on_continuous_value([np.NINF, np.NINF, 0.2, 0.7, 1.2, 4.3])(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.INT64, [None, 1])
    
    assert_model_result(g, 
        input={
            'i': [
                [1.3],
                [0.6999],
                [-9.6],
                [9.6],
            ]
        },
        expected_result=[[
            [4],
            [2],
            [1],
            [5],
        ]]
    )


def test_get_bin_index_on_categorical_value():
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.STRING, [None, 1])

    g = ebm.get_bin_index_on_categorical_value({
        'foo': 1,
        'bar': 2,
        'biz': 3,
    })(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.INT64, [None, 1])

    result = infer_model(graph.compile(g, target_opset=13),
        input={
            'i': [["biz"], ["foo"], ["bar"], ["nan"], ["okif"]],
        }
    )
    expected_result=[
            [3],
            [1],
            [2],
            [0],
            [-1],
        ]
    assert len(expected_result) == len(result[0])
    for i, r in enumerate(expected_result):
        assert result[0][i] == r


def test_get_bin_score_1d():
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.INT64, [None, 1])

    g = ebm.get_bin_score_1d(np.array([0.0, 0.1, 0.2, 0.3]))(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, 1, 1])
    
    assert_model_result(g, 
        input={
            'i': [
                [3],
                [1],
                [2],
                [0],
            ]
        },
        expected_result=[[
            [[0.3]],
            [[0.1]],
            [[0.2]],
            [[0.0]],
        ]]
    )


def test_get_bin_score_1d_multiclass():
    """test on 3 classes
    shape of scores is [bin_count x class_count]
    """
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.INT64, [None, 1])

    g = ebm.get_bin_score_1d(np.array(

        [
            [0.0, 1.0, 2.0],
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
            [0.3, 1.3, 2.3],
        ]
    ))(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, 1, 3])

    assert_model_result(g,
        input={
            'i': [
                [3],
                [1],
                [2],
                [0],
                [2],
            ]
        },
        expected_result=[[
            [[0.3, 1.3, 2.3]],
            [[0.1, 1.1, 2.1]],
            [[0.2, 1.2, 2.2]],
            [[0.0, 1.0, 2.0]],
            [[0.2, 1.2, 2.2]],
        ]],
    )

def test_get_bin_score_2d():
    g = graph.create_graph()
    i1 = graph.create_input(g, "i1", onnx.TensorProto.INT64, [None, 1])
    i2 = graph.create_input(g, "i2", onnx.TensorProto.INT64, [None, 1])

    i = graph.merge(i1, i2)
    g = ebm.get_bin_score_2d(np.array([
        [0.0, 0.1, 0.2, 0.3],
        [1.0, 2.1, 3.2, 4.3],
        [10.0, 20.1, 30.2, 40.3],
    ]))(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, 1, 1])
    
    assert_model_result(g, 
        input={
            'i1': [[2], [1], [2], [0]],
            'i2': [[3], [0], [2], [1]],
        },
        expected_result=[[
            [[40.3]],
            [[1.0]],
            [[30.2]],
            [[0.1]],
        ]]
    )


def test_compute_class_score():
    g = graph.create_graph()
    i1 = graph.create_input(g, "i1", onnx.TensorProto.FLOAT, [None, 1, 1])
    i2 = graph.create_input(g, "i2", onnx.TensorProto.FLOAT, [None, 1, 1])
    i3 = graph.create_input(g, "i3", onnx.TensorProto.FLOAT, [None, 1, 1])

    i = graph.merge(i1, i2, i3)
    g, _ = ebm.compute_class_score(np.array([0.2]), explain_name="scores")(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, 1])
    
    assert_model_result(g, 
        input={
            'i1': [[[0.1]], [[0.2]], [[0.3]], [[0.4]]],
            'i2': [[[1.1]], [[1.2]], [[1.3]], [[1.4]]],
            'i3': [[[2.1]], [[2.2]], [[2.3]], [[2.4]]],
        },
        expected_result=[[
            [3.5],
            [3.8],
            [4.1],
            [4.4],
        ]]
    )


def test_compute_multiclass_score():
    g = graph.create_graph()
    i1 = graph.create_input(g, "i1", onnx.TensorProto.FLOAT, [None, 1, 3])
    i2 = graph.create_input(g, "i2", onnx.TensorProto.FLOAT, [None, 1, 3])
    i3 = graph.create_input(g, "i3", onnx.TensorProto.FLOAT, [None, 1, 3])

    i = graph.merge(i1, i2, i3)
    g, _ = ebm.compute_class_score(np.array([0.1, 0.2, 0.3]), explain_name="scores")(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, 3])

    assert_model_result(g,
        input={
            'i1': [
                [[0.1, 0.2, 0.3]],
                [[0.2, 0.3, 0.4]],
                [[0.3, 0.4, 0.5]],
                [[0.4, 0.5, 0.6]]],
            'i2': [
                [[1.1, 1.2, 1.3]],
                [[1.2, 1.3, 1.4]],
                [[1.3, 1.4, 1.5]],
                [[1.4, 1.5, 1.6]]],
            'i3': [
                [[2.1, 2.2, 2.3]],
                [[2.2, 2.3, 2.4]],
                [[2.3, 2.4, 2.5]],
                [[2.4, 2.5, 2.6]]],
        },
        expected_result=[[
            [3.4, 3.8, 4.2],
            [3.7, 4.1, 4.5],
            [4.0, 4.4, 4.8],
            [4.3, 4.7, 5.1],
        ]]
    )


def test_predict_class_binary():
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 1])

    g = ebm.predict_class(
        classes=[0, 1], class_type=onnx.TensorProto.INT64,
        binary=True, prediction_name="prediction"
    )(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.INT64, [None])
    
    assert_model_result(g, 
        input={
            'i': [[3.5], [-3.8], [-0.1], [0.2]]
        },
        expected_result=[[1, 0, 0, 1]]
    )


def test_predict_multiclass_binary():
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 3])

    g = ebm.predict_class(
        classes=[0, 1, 2], class_type=onnx.TensorProto.INT64,
        binary=False, prediction_name="prediction"
    )(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.INT64, [None])

    assert_model_result(g,
        input={
            'i': [
            [3.4, 3.8, 4.2],
            [3.7, 4.1, 0.5],
            [4.0, 0.4, 0.8],
            [4.3, 4.7, 5.1],
        ]
        },
        expected_result=[[2, 1, 0, 2]]
    )


def test_predict_proba_binary():
    g = graph.create_graph()
    i = graph.create_input(g, "i", onnx.TensorProto.FLOAT, [None, 1])

    g = ebm.predict_proba(binary=True, probabilities_name="probabilities")(i)
    g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, 2])
    
    assert_model_result(g, 
        input={
            'i': [[3.5], [-3.8], [-0.1], [0.2]]
        },
        expected_result=[[
            [0.02931223, 0.97068775],
            [0.97811866, 0.02188127],
            [0.5249792 , 0.4750208 ],
            [0.450166  , 0.54983395],
        ]]
    )

