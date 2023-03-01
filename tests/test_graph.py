import ebm2onnx.graph as graph
import onnx


def test_create_graph():
    g = graph.create_graph()

    assert g.generate_name is not None
    assert g.inputs == []
    assert g.outputs == []
    assert g.nodes == []


def test_create_one_input():
    g = graph.create_graph()

    input = graph.create_input(g, "foo", onnx.TensorProto.FLOAT, [None, 3])
    assert len(input.inputs) == 1
    assert input.inputs == [onnx.helper.make_tensor_value_info(
        'foo' ,
        onnx.TensorProto.FLOAT,
        [None, 3])
    ]
    assert input.inputs == input.transients


def test_create_several_inputs():
    g = graph.create_graph()

    i1 = graph.create_input(g, "foo", onnx.TensorProto.FLOAT, [None, 3])
    i2 = graph.create_input(g, "bar", onnx.TensorProto.INT64, [None, 2])

    assert i1.inputs == [onnx.helper.make_tensor_value_info(
        'foo' ,
        onnx.TensorProto.FLOAT,
        [None, 3])
    ]
    assert i1.inputs == i1.transients

    assert i2.inputs == [onnx.helper.make_tensor_value_info(
        f'bar' ,
        onnx.TensorProto.INT64,
        [None, 2])
    ]
    assert i2.inputs == i2.transients


def test_create_initializer():
    g = graph.create_graph()

    init = graph.create_initializer(g, "foo", onnx.TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4])
    assert len(init.initializers) == 1
    assert init.initializers == [onnx.helper.make_tensor(
        'foo_0' ,
        onnx.TensorProto.FLOAT,
        [4],
        [0.1, 0.2, 0.3, 0.4]
    )]
    assert init.initializers == init.transients


def test_merge():
    g = graph.create_graph()

    init1 = graph.create_initializer(g, "foo", onnx.TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4])
    init2 = graph.create_initializer(g, "foo", onnx.TensorProto.FLOAT, [4], [1.1, 1.2, 3.3, 4.4])
    input1 = graph.create_input(g, "bar1", onnx.TensorProto.FLOAT, [None, 3])
    input2 = graph.create_input(g, "bar2", onnx.TensorProto.FLOAT, [None, 4])

    m = graph.merge(init1, input1, init2, input2)

    assert len(m.initializers) == 2
    assert len(m.inputs) == 2
    assert len(m.transients) == 4

    assert m.initializers == [
        onnx.helper.make_tensor(
            'foo_0' ,
            onnx.TensorProto.FLOAT,
            [4],
            [0.1, 0.2, 0.3, 0.4]
        ),
        onnx.helper.make_tensor(
            'foo_1' ,
            onnx.TensorProto.FLOAT,
            [4],
            [1.1, 1.2, 3.3, 4.4]
        ),
    ]

    assert m.inputs == [
        onnx.helper.make_tensor_value_info(
            'bar1' ,
            onnx.TensorProto.FLOAT,
            [None, 3],
        ),
        onnx.helper.make_tensor_value_info(
            'bar2' ,
            onnx.TensorProto.FLOAT,
            [None, 4],
        ),
    ]

    assert m.transients == [
        onnx.helper.make_tensor(
            'foo_0' ,
            onnx.TensorProto.FLOAT,
            [4],
            [0.1, 0.2, 0.3, 0.4]
        ),
        onnx.helper.make_tensor_value_info(
            'bar1' ,
            onnx.TensorProto.FLOAT,
            [None, 3],
        ),
        onnx.helper.make_tensor(
            'foo_1' ,
            onnx.TensorProto.FLOAT,
            [4],
            [1.1, 1.2, 3.3, 4.4]
        ),
        onnx.helper.make_tensor_value_info(
            'bar2' ,
            onnx.TensorProto.FLOAT,
            [None, 4],
        ),
    ]


def test_strip_to_transients():
    g = graph.create_graph()

    input1 = graph.create_input(g, "bar1", onnx.TensorProto.FLOAT, [None, 3])
    input2 = graph.create_input(g, "bar2", onnx.TensorProto.FLOAT, [None, 4])

    m = graph.merge(input1, input2)
    m = graph.strip_to_transients(m)

    assert m.initializers == []
    assert m.inputs == []
    assert m.transients == [
        onnx.helper.make_tensor_value_info(
            'bar1' ,
            onnx.TensorProto.FLOAT,
            [None, 3],
        ),
        onnx.helper.make_tensor_value_info(
            'bar2' ,
            onnx.TensorProto.FLOAT,
            [None, 4],
        ),
    ]


def test_clear_transients():
    g = graph.create_graph()

    input1 = graph.create_input(g, "bar1", onnx.TensorProto.FLOAT, [None, 3])
    input2 = graph.create_input(g, "bar2", onnx.TensorProto.FLOAT, [None, 4])

    m = graph.merge(input1, input2)
    m = graph.clear_transients(m)

    assert m.initializers == []
    assert m.inputs == [
        onnx.helper.make_tensor_value_info(
            'bar1' ,
            onnx.TensorProto.FLOAT,
            [None, 3],
        ),
        onnx.helper.make_tensor_value_info(
            'bar2' ,
            onnx.TensorProto.FLOAT,
            [None, 4],
        ),
    ]
    assert m.transients == []
