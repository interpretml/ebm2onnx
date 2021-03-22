import onnx
import ebm2onnx.graph as graph


def add():
    def _add(g):
        add_result_name = g.generate_name('add_result')
        nodes = [
            onnx.helper.make_node("Add", [g.transients[0].name, g.transients[1].name], [add_result_name]),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(add_result_name, g.transients[0].type.tensor_type.elem_type, []),
            ],
        )

    return _add


def argmax(axis=0, keepdims=1, select_last_index=0):
    def _argmax(g):        
        argmax_result_name = g.generate_name('argmax_result')
        nodes = [
            onnx.helper.make_node(
                "ArgMax",
                [g.transients[0].name], [argmax_result_name],
                axis=axis, keepdims=keepdims, select_last_index=select_last_index
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(argmax_result_name, onnx.TensorProto.INT64, []),
            ],
        )

    return _argmax


def cast(to):
    def _cast(g):        
        cast_result_name = g.generate_name('cast_result')
        nodes = [
            onnx.helper.make_node("Cast", [g.transients[0].name], [cast_result_name], to=to),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(cast_result_name, to, []),
            ],
        )

    return _cast


def concat(axis):
    def _concat(g):
        concat_result_name = g.generate_name('concat_result')

        sources = [t.name for t in g.transients]
        nodes = [
            onnx.helper.make_node("Concat", sources, [concat_result_name], axis=axis),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(concat_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _concat


def expand():
    def _expand(g):        
        expand_result_name = g.generate_name('expand_result')
        nodes = [
            onnx.helper.make_node("Expand", [g.transients[0].name, g.transients[1].name], [expand_result_name]),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(expand_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _expand



def flatten(axis=1):
    def _flatten(g):        
        flatten_result_name = g.generate_name('flatten_result')
        nodes = [
            onnx.helper.make_node("Flatten", [g.transients[0].name], [flatten_result_name], axis=axis),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(flatten_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _flatten


def gather_elements(axis=0):
    def _gather_elements(g):
        gather_elements_result_name = g.generate_name('gather_elements_result')        
        nodes = [
            onnx.helper.make_node(
                "GatherElements",
                [g.transients[0].name, g.transients[1].name],
                [gather_elements_result_name],
                axis=axis,
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(gather_elements_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _gather_elements


def gather_nd():
    """
    Input transients:
        - scores, as a 2D matrix
        - indices, as a [None, 2] matrix
    """
    def _gather_nd(g):
        gather_nd_result_name = g.generate_name('gather_nd_result')        
        nodes = [
            onnx.helper.make_node("GatherND", [g.transients[0].name, g.transients[1].name], [gather_nd_result_name]),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(gather_nd_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _gather_nd


def identity(name):
    def _identity(g):
        identity_name = g.generate_name(name)
        nodes = [
            onnx.helper.make_node("Identity", [g.transients[0].name], [identity_name]),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(identity_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _identity


def less():
    def _less(g):        
        less_result_name = g.generate_name('less_result')        
        nodes = [
            onnx.helper.make_node("Less", [g.transients[0].name, g.transients[1].name], [less_result_name]),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(less_result_name, onnx.TensorProto.BOOL, []),
            ],
        )

    return _less


def less_or_equal():
    def _less_or_equal(g):        
        less_or_equal_result_name = g.generate_name('less_or_equal_result')
        nodes = [
            onnx.helper.make_node("LessOrEqual", [g.transients[0].name, g.transients[1].name], [less_or_equal_result_name]),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(less_or_equal_result_name, onnx.TensorProto.BOOL, []),
            ],
        )

    return _less_or_equal

def mul():
    def _mul(g):        
        mul_result_name = g.generate_name('mul_result')        
        nodes = [
            onnx.helper.make_node("Mul", [g.transients[0].name, g.transients[1].name], [mul_result_name]),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(mul_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )


    return _mul


def reduce_sum(keepdims=1, noop_with_empty_axes=0):
    def _reduce_sum(g):        
        reduce_sum_result_name = g.generate_name('reduce_sum_result')        
        nodes = [
            onnx.helper.make_node(
                "ReduceSum",
                [g.transients[0].name, g.transients[1].name],
                [reduce_sum_result_name],
                keepdims=keepdims,
                noop_with_empty_axes=noop_with_empty_axes,
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(reduce_sum_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )


    return _reduce_sum


def reshape(allowzero=0):
    def _reshape(g):        
        reshape_result_name = g.generate_name('reshape_result')        
        nodes = [
            onnx.helper.make_node(
                "Reshape",
                [g.transients[0].name, g.transients[1].name],
                [reshape_result_name],
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(reshape_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _reshape


def softmax(axis=-1):
    def _softmax(g):
        softmax_result_name = g.generate_name('softmax_result')        
        nodes = [
            onnx.helper.make_node(
                "Softmax",
                [g.transients[0].name],
                [softmax_result_name],
                axis=axis,
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(softmax_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _softmax
