import onnx
import ebm2onnx.operators as ops
import ebm2onnx.graph as graph


def get_bin_index_on_continuous_value(bin_edges):
    """

    Input graph
    """
    def _get_bin_index_on_continuous_value(g):
        bin_count = len(bin_edges)
        index_range = list(range(bin_count))
        
        init_bin_index_range = graph.create_initializer(g, "bin_index_range", onnx.TensorProto.FLOAT, [bin_count], index_range)
        init_bin_edges = graph.create_initializer(g, "bin_edges", onnx.TensorProto.DOUBLE, [bin_count], bin_edges)

        g = ops.cast(onnx.TensorProto.DOUBLE)(g)
        g = ops.less_or_equal()(graph.merge(init_bin_edges, g))
        g = ops.cast(onnx.TensorProto.FLOAT)(g)
        g = ops.mul()(graph.merge(g, init_bin_index_range))
        g = ops.argmax(axis=1)(g)
        return g

    return _get_bin_index_on_continuous_value


def get_bin_score_1d(bin_scores):
    def _get_bin_score_1d(g):
        init_bin_scores = graph.create_initializer(g, "bin_scores", onnx.TensorProto.FLOAT, [len(bin_scores), 1], bin_scores)

        g = ops.gather_elements()(graph.merge(init_bin_scores, g))
        return g

    return _get_bin_score_1d


def get_bin_score_2d(bin_scores):
    def _get_bin_score_2d(g):
        init_bin_scores = graph.create_initializer(
            g, "bin_scores", onnx.TensorProto.FLOAT,
            [bin_scores.shape[0], bin_scores.shape[1]],
            bin_scores.flatten(),
        )

        g = ops.concat(axis=1)(g)
        g = ops.gather_nd()(graph.merge(init_bin_scores, g))
        g = ops.flatten()(g)
        return g

    return _get_bin_score_2d


def compute_class_score(intercept):
    def _compute_class_score(g):
        init_intercept = graph.create_initializer(
            g, "intercept", onnx.TensorProto.FLOAT,
            [1], [intercept],
        )
        init_sum_axis = graph.create_initializer(
            g, "intercept", onnx.TensorProto.INT64,
            [1], [1],
        )

        g = ops.concat(axis=1)(g)
        scores_output_name = g.transients[0].name
        g = ops.reduce_sum()(graph.merge(g, init_sum_axis))
        g = ops.add()(graph.merge(g, init_intercept))
        return g, scores_output_name

    return _compute_class_score


def predict_class(binary):
    def _predict_class(g):
        if binary is True:
            init_zeros = graph.create_initializer(
                g, "zeros", onnx.TensorProto.FLOAT,
                [2], [0.0, 1.0],
            )

            init_reshape = graph.create_initializer(
                g, "reshape", onnx.TensorProto.INT64,
                [1], [0],
            )

            g = ops.mul()(graph.merge(g, init_zeros))
        g = ops.argmax(axis=1)(g)
        g = ops.reshape()(graph.merge(g, init_reshape))
        return g

    return _predict_class


def predict_proba(binary):
    def _predict_proba(g):
        if binary is True:
            init_zeros = graph.create_initializer(
                g, "zeros", onnx.TensorProto.FLOAT,
                [2], [0.0, 1.0],
            )

            init_reshape = graph.create_initializer(
                g, "reshape", onnx.TensorProto.INT64,
                [1], [0],
            )

            g = ops.mul()(graph.merge(g, init_zeros))
        g = ops.softmax(axis=1)(g)
        return g

    return _predict_proba
