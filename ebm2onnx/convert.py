from collections import namedtuple
from .utils import get_latest_opset_version
from ebm2onnx import graph
from ebm2onnx import ebm
import ebm2onnx.operators as ops

import numpy as np
import onnx

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


onnx_type_for={
    'bool': onnx.TensorProto.BOOL,
    'float': onnx.TensorProto.FLOAT,
    'double': onnx.TensorProto.DOUBLE,
    'int': onnx.TensorProto.INT64,
    'str': onnx.TensorProto.STRING,
}


def infer_features_dtype(dtype, feature_name):
    feature_dtype = onnx.TensorProto.DOUBLE
    if dtype is not None:
        feature_dtype = onnx_type_for[dtype[feature_name]]

    return feature_dtype


def get_dtype_from_pandas(df):
    """Infers the features names and types from a pandas dataframe

    Example:
        >>>import ebm2onnx
        >>>
        >>>dtype = ebm2onnx.get_dtype_from_pandas(my_df)
    Args:
        df: A pandas dataframe

    Returns:
        A dict that can be used as the type argument of the to_onnx function.
    """
    dtype = {}
    df_types = df.dtypes.values
    for i, k in enumerate(df.dtypes.index):
        if df_types[i] == np.float32:
            dtype[k] = 'float'
        elif df_types[i] == np.double:
            dtype[k] = 'double'
        elif df_types[i] == int:
            dtype[k] = 'int'
        elif df_types[i] == bool:
            dtype[k] = 'bool'
        elif df_types[i] == str:
            dtype[k] = 'str'
        elif df_types[i] == object:
            dtype[k] = 'str'
        else:
            raise ValueError("column {} is of type {} that is not supported".format(k, df_types[i]))

    return dtype


def to_onnx(model, dtype, name="ebm",
            predict_proba=False,
            explain=False,
            target_opset=None,
            ):
    """Converts an EBM model to ONNX

    Args:
        model: The EBM model, trained with interpretml
        dtype: A dict containing the type of each input feature. Types are expressed as strings, the following values are supported: float, double, int, str.
        name: [Optional] The name of the model
        predict_proba: [Optional] For classification models, output prediction probabilities instead of class
        explain: [Optional] Adds an additional output with the score per feature per class
        target_opset: [Optional] The target onnx opset version to use

    Returns:
        An ONNX model.
    """
    target_opset = target_opset or get_latest_opset_version()
    root = graph.create_graph()

    class_index=0
    inputs = [None for _ in model.feature_names_in_]
    parts = []

    feature_types = list(model.feature_types_in_)
    interaction_count = len(model.term_names_) - len(feature_types)
    for _ in range(interaction_count):
        feature_types.append('interaction')

    # first compute the score of each feature
    for feature_index in range(len(model.term_names_)):
        feature_name=model.term_names_[feature_index]
        feature_type=feature_types[feature_index]
        feature_group=model.term_features_[feature_index]

        if feature_type == 'continuous':
            bins = [np.NINF, np.NINF] + list(model.bins_[feature_group[0]][0])
            additive_terms = model.term_scores_[feature_index]

            feature_dtype = infer_features_dtype(dtype, feature_name)
            part = graph.create_input(root, feature_name, feature_dtype, [None])
            part = ops.flatten()(part)
            inputs[feature_index] = part
            part = ebm.get_bin_index_on_continuous_value(bins)(part)
            part = ebm.get_bin_score_1d(additive_terms)(part)
            parts.append(part)

        elif feature_type in ['nominal', 'ordinal']:
            col_mapping = model.bins_[feature_group[0]][0]
            additive_terms = model.term_scores_[feature_index]

            feature_dtype = infer_features_dtype(dtype, feature_name)
            part = graph.create_input(root, feature_name, feature_dtype, [None])
            if feature_dtype != onnx.TensorProto.STRING:
                part = ops.cast(onnx.TensorProto.STRING)(part)
            part = ops.flatten()(part)
            inputs[feature_index] = part
            part = ebm.get_bin_index_on_categorical_value(col_mapping)(part)
            part = ebm.get_bin_score_1d(additive_terms)(part)
            parts.append(part)

        elif feature_type == 'interaction':
            i_parts = []
            way_count = len(feature_group)

            for index in range(way_count):
                i_feature_index = feature_group[index]
                i_feature_type = feature_types[i_feature_index]

                if i_feature_type == 'continuous':
                    # interactions can be of any size (n way).
                    # There may be one binning per interaction way or not.
                    # the rule is to use bins_ index if there is one binning available for the way count.
                    # otherwise, use the last binning for the feature
                    bin_index = -1 if way_count > len(model.bins_[i_feature_index]) else way_count - 1
                    bins = [np.NINF, np.NINF] + list(model.bins_[i_feature_index][bin_index])
                    input = graph.strip_to_transients(inputs[i_feature_index])
                    i_parts.append(ebm.get_bin_index_on_continuous_value(bins)(input))

                elif i_feature_type in ['nominal', 'ordinal']:
                    col_mapping = model.bins_[i_feature_index][0]
                    input = graph.strip_to_transients(inputs[i_feature_index])
                    i_parts.append(ebm.get_bin_index_on_categorical_value(col_mapping)(input))

                else:
                    raise ValueError(f"The type of the feature {feature_name} is unknown: {feature_type}")

            part = graph.merge(*i_parts)
            additive_terms = model.term_scores_[feature_index]
            part = ebm.get_bin_score_2d(np.array(additive_terms))(part)
            parts.append(part)

        else:
            raise ValueError(f"The type of the feature {feature_name} is unknown: {feature_type}")

    # compute scores, predict and proba
    g = graph.merge(*parts)
    if type(model) is ExplainableBoostingClassifier:
        g, scores_output_name = ebm.compute_class_score(model.intercept_)(g)
        if len(model.classes_) == 2: # binary classification
            if predict_proba is False:
                g = ebm.predict_class(binary=True)(g)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.INT64, [None])
            else:
                g = ebm.predict_proba(binary=True)(g)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, len(model.classes_)])
        else:
            if predict_proba is False:
                g = ebm.predict_class(binary=False)(g)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.INT64, [None])
            else:
                g = ebm.predict_proba(binary=False)(g)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, len(model.classes_)])

        if explain is True:
            if len(model.classes_) == 2:
                g = graph.add_output(g, scores_output_name, onnx.TensorProto.FLOAT, [None, len(model.term_names_), 1])
            else:
                g = graph.add_output(g, scores_output_name, onnx.TensorProto.FLOAT, [None, len(model.term_names_), len(model.classes_)])
    elif type(model) is ExplainableBoostingRegressor:
        g, scores_output_name = ebm.compute_class_score(np.array([model.intercept_]))(g)
        g = ebm.predict_value()(g)
        g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None])
        g = graph.add_output(g, scores_output_name, onnx.TensorProto.FLOAT, [None, len(model.term_names_), 1])
    else:
        raise NotImplementedError("{} models are not supported".format(type(model)))



    model = graph.compile(g, target_opset, name=name)
    return model
