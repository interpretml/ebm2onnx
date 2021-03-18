import onnx
import ebm2onnx.graph as graph


def category_mapper(cats_int64s, cats_strings, default_int64=-1, default_string="_Unused"):
    def _category_mapper(g):
        category_mapper_result_name = g.generate_name('category_mapper_result')
        nodes = [
            onnx.helper.make_node(
                "CategoryMapper",
                [g.transients[0].name], [category_mapper_result_name],
                cats_int64s=cats_int64s,
                cats_strings=cats_strings,
                default_int64=default_int64,
                default_string=default_string,
                domain='ai.onnx.ml',
            ),
        ]

        return g._replace(
            nodes=graph.extend(g.nodes, nodes),
            transients=[
                 onnx.helper.make_tensor_value_info(category_mapper_result_name, onnx.TensorProto.UNDEFINED, []),
            ],
        )

    return _category_mapper
