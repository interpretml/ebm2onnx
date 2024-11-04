import onnx
import ebm2onnx.graph as graph


def add():
    def _add(g):
        add_result_name = g.context.generate_variable_name('add_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Add",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[add_result_name],
                name=g.context.generate_operator_name('Add'),
            ),
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
        argmax_result_name = g.context.generate_variable_name('argmax_result')
        nodes = [
            onnx.helper.make_node(
                op_type="ArgMax",
                inputs=[g.transients[0].name],
                outputs=[argmax_result_name],
                name=g.context.generate_operator_name('ArgMax'),
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
        cast_result_name = g.context.generate_variable_name('cast_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Cast",
                inputs=[g.transients[0].name],
                outputs=[cast_result_name],
                name=g.context.generate_operator_name('Cast'),
                to=to,
            ),
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
        concat_result_name = g.context.generate_variable_name('concat_result')

        sources = [t.name for t in g.transients]
        nodes = [
            onnx.helper.make_node(
                op_type="Concat",
                inputs=sources,
                outputs=[concat_result_name],
                name=g.context.generate_operator_name('Concat'),
                axis=axis,
            ),
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
        expand_result_name = g.context.generate_variable_name('expand_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Expand",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[expand_result_name],
                name=g.context.generate_operator_name('Expand'),
            ),
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
        flatten_result_name = g.context.generate_variable_name('flatten_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Flatten",
                inputs=[g.transients[0].name],
                outputs=[flatten_result_name],
                name=g.context.generate_operator_name('Flatten'),
                axis=axis,
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                onnx.helper.make_tensor_value_info(flatten_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _flatten


def gather(axis=0):
    """
    Input transients:
        - data
        - indices
    """
    def _gather(g):
        gather_result_name = g.context.generate_variable_name('gather_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Gather",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[gather_result_name],
                name=g.context.generate_operator_name('Gather'),
                axis=axis,
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                onnx.helper.make_tensor_value_info(gather_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _gather


def gather_elements(axis=0):
    def _gather_elements(g):
        gather_elements_result_name = g.context.generate_variable_name('gather_elements_result')
        nodes = [
            onnx.helper.make_node(
                op_type="GatherElements",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[gather_elements_result_name],
                name=g.context.generate_operator_name('GatherElements'),
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
        gather_nd_result_name = g.context.generate_variable_name('gather_nd_result')
        nodes = [
            onnx.helper.make_node(
                "GatherND",
                [g.transients[0].name, g.transients[1].name],
                [gather_nd_result_name],
                name=g.context.generate_operator_name('GatherND'),
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                onnx.helper.make_tensor_value_info(gather_nd_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _gather_nd


def greater_or_equal():
    def _greater_or_equal(g):
        greater_or_equal_result_name = g.context.generate_variable_name('greater_or_equal_result')
        nodes = [
            onnx.helper.make_node(
                op_type="GreaterOrEqual",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[greater_or_equal_result_name],
                name=g.context.generate_operator_name('GreaterOrEqual'),
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                onnx.helper.make_tensor_value_info(greater_or_equal_result_name, onnx.TensorProto.BOOL, []),
            ],
        )

    return _greater_or_equal


def identity(name, suffix=True):
    def _identity(g):
        identity_name = g.context.generate_variable_name(name) if suffix else name
        nodes = [
            onnx.helper.make_node(
                op_type="Identity",
                inputs=[g.transients[0].name],
                outputs=[identity_name],
                name=g.context.generate_operator_name('Identity'),
            ),
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
        less_result_name = g.context.generate_variable_name('less_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Less",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[less_result_name],
                name=g.context.generate_operator_name('Less'),
            ),
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
        less_or_equal_result_name = g.context.generate_variable_name('less_or_equal_result')
        nodes = [
            onnx.helper.make_node(
                "LessOrEqual",
                [g.transients[0].name, g.transients[1].name],
                [less_or_equal_result_name],
                name=g.context.generate_operator_name('LessOrEqual'),
            ),
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
        mul_result_name = g.context.generate_variable_name('mul_result')
        nodes = [
            onnx.helper.make_node(
                "Mul",
                [g.transients[0].name, g.transients[1].name],
                [mul_result_name],
                name=g.context.generate_operator_name('Mul'),
            ),
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
        reduce_sum_result_name = g.context.generate_variable_name('reduce_sum_result')
        nodes = [
            onnx.helper.make_node(
                op_type="ReduceSum",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[reduce_sum_result_name],
                name=g.context.generate_operator_name('ReduceSum'),
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
        reshape_result_name = g.context.generate_variable_name('reshape_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Reshape",
                inputs=[g.transients[0].name, g.transients[1].name],
                outputs=[reshape_result_name],
                name=g.context.generate_operator_name('Reshape'),
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
        softmax_result_name = g.context.generate_variable_name('softmax_result')
        nodes = [
            onnx.helper.make_node(
                op_type="Softmax",
                inputs=[g.transients[0].name],
                outputs=[softmax_result_name],
                name=g.context.generate_operator_name('Softmax'),
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


def split(axis=0):
    def _split(g):
        split_result_name = [
            g.context.generate_variable_name('split_result')
            for _ in range(list(g.transients[0].type.tensor_type.shape.dim)[axis].dim_value)
        ]

        nodes = [
            onnx.helper.make_node(
                op_type="Split",
                inputs=[g.transients[0].name],
                outputs=split_result_name,
                name=g.context.generate_operator_name('Split'),
                axis=axis,
                num_outputs=len(split_result_name),
            ),  
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UNDEFINED, [])
                for name in split_result_name
            ]
        )

    return _split
