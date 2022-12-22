import numpy as np
import onnx
import ebm2onnx.operators as ops
import ebm2onnx.operators_ml as mlops
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


def get_bin_index_on_categorical_value(col_mapping, missing_str=str(np.nan)):
    def _get_bin_index_on_categorical_value(g):
        ints = [0]
        strings = [missing_str]
        for k, v in col_mapping.items():
            ints.append(v)
            strings.append(k)

        g = mlops.category_mapper(
            cats_int64s=ints,
            cats_strings=strings,
        )(g)
        g = ops.flatten()(g)
        return g

    return _get_bin_index_on_categorical_value


def get_bin_score_1d(bin_scores):
    if len(bin_scores.shape) == 1:
        bin_scores= bin_scores.reshape((-1, 1))

    def _get_bin_score_1d(g):
        init_bin_scores = graph.create_initializer(
            g, "bin_scores", onnx.TensorProto.FLOAT,
            bin_scores.shape, bin_scores.flatten())

        init_reshape = graph.create_initializer(
            g, "score_reshape", onnx.TensorProto.INT64,
            [3], [-1, 1, bin_scores.shape[1]],
        )

        g = ops.gather_nd()(graph.merge(init_bin_scores, g))  # gather score for each class
        g = ops.reshape()(graph.merge(g, init_reshape))
        return g

    return _get_bin_score_1d


def get_bin_score_2d(bin_scores):
    def _get_bin_score_2d(g):
        init_bin_scores = graph.create_initializer(
            g, "bin_scores", onnx.TensorProto.FLOAT,
            bin_scores.shape,
            bin_scores.flatten(),
        )

        init_reshape = graph.create_initializer(
            g, "score_reshape", onnx.TensorProto.INT64,
            [3], [-1, 1, 1],
        )

        g = ops.concat(axis=1)(g)
        g = ops.gather_nd()(graph.merge(init_bin_scores, g))
        g = ops.reshape()(graph.merge(g, init_reshape))
        return g

    return _get_bin_score_2d


def compute_class_score(intercept):
    """
    intercept shape: [class_count]
    input shapes: [batch_size x 1 x class_count]
    output shape: [batch_size x class_count]

    score output shape: [batch_size x feature_count x class_count]
    """
    def _compute_class_score(g):
        init_intercept = graph.create_initializer(
            g, "intercept", onnx.TensorProto.FLOAT,
            [intercept.shape[0]], intercept,
        )
        init_sum_axis = graph.create_initializer(
            g, "sum_axis", onnx.TensorProto.INT64,
            [1], [1],
        )

        g = ops.concat(axis=1)(g)
        g = ops.identity("scores")(g)
        scores_output_name = g.transients[0].name
        g = ops.reduce_sum(keepdims=0)(graph.merge(g, init_sum_axis))
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

            g = ops.mul()(graph.merge(g, init_zeros))

        init_reshape = graph.create_initializer(
            g, "reshape", onnx.TensorProto.INT64,
            [1], [0],
        )

        g = ops.argmax(axis=1)(g)
        g = ops.reshape()(graph.merge(g, init_reshape))
        g = ops.identity("predict")(g)
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
        g = ops.identity("predict_proba")(g)
        return g

    return _predict_proba


def predict_value():
    """Final prediction step for regression

    No operations are needed here, we just reshape the
    [None, 1] scores to [None].
    """
    def _predict_value(g):
        init_reshape = graph.create_initializer(
            g, "reshape", onnx.TensorProto.INT64,
            [1], [0],
        )

        g = ops.reshape()(graph.merge(g, init_reshape))
        g = ops.identity("predict")(g)
        return g

    return _predict_value
