from typing import List
import logging
from enum import Enum
from copy import deepcopy
from ebm2onnx import graph
from ebm2onnx import ebm
import ebm2onnx.operators as ops

import numpy as np
import onnx

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


onnx_type_for = {
    'bool': onnx.TensorProto.BOOL,
    'float': onnx.TensorProto.FLOAT,
    'double': onnx.TensorProto.DOUBLE,
    'int': onnx.TensorProto.INT64,
    'str': onnx.TensorProto.STRING,
}

np_type_for = {
    'bool': bool,
    'float': np.float32,
    'double': np.double,
    'int': int,
    'str': str,
}

bool_remap = {
    'False': '0',
    'True': '1',
}


class FeatureType(Enum):
    COLUMN = 1
    TENSOR = 2


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


def get_dtype_from_tensor_type(
    dtype: str,
    features: List[str]
):
    return {
        f: dtype
        for f in features
    }


def to_graph(model, dtype, name="ebm",
            predict_proba=False,
            explain=False,
            target_opset=None,
            prediction_name="prediction",
            probabilities_name="probabilities",
            explain_name="scores",
            context=None,
            ):
    """Converts an EBM model to a graph.

    The returned model contains one to three output.
    The first output is always the prediction, and is named "prediction".
    If predict_proba is set to True, then another output named "probabilities" is added.
    If explain is set to True, then another output named "scores" is added.

    Args:
        model: The EBM model, trained with interpretml
        dtype: A dict containing the type of each input feature. Types are expressed as strings, the following values are supported: float, double, int, str.
        name: [Optional] The name of the model
        predict_proba: [Optional] For classification models, output prediction probabilities instead of class
        explain: [Optional] Adds an additional output with the score per feature per class
        target_opset: [Optional][Deprecated] The target onnx opset version to use

    Returns:
        An ONNX model.
    """
    if target_opset:
        logging.warning("to_graph: target_opset argument is deprecated")
    root = graph.create_graph(context=context)

    inputs = [None for _ in model.feature_names_in_]
    parts = []

    if type(dtype) is tuple:
        dname, dtype = dtype
        logging.debug(f"using tensor-based input {dtype} of len {len(model.feature_names_in_)}")
        features_org = FeatureType.TENSOR
        tensor_inputs = graph.create_input(root, dname, onnx_type_for[dtype], [None, len(model.feature_names_in_)])
        tensor_inputs = ebm.split_input(model.feature_names_in_)(tensor_inputs)
        tensor_inputs = graph.clear_transients(tensor_inputs)
        dtype = get_dtype_from_tensor_type(dtype, model.feature_names_in_)
    else:
        logging.debug(f"using column-based inputs {model.feature_names_in_}")
        features_org = FeatureType.COLUMN

    feature_types = list(model.feature_types_in_)
    interaction_count = len(model.term_names_) - len(feature_types)
    for _ in range(interaction_count):
        feature_types.append('interaction')

    model_bins = deepcopy(model.bins_)

    # first compute the score of each feature
    for feature_index in range(len(model.term_names_)):
        feature_name = model.term_names_[feature_index]
        feature_type = feature_types[feature_index]
        feature_group = model.term_features_[feature_index]

        if feature_type == 'continuous':
            bins = [-np.inf, -np.inf] + list(model_bins[feature_group[0]][0])
            additive_terms = model.term_scores_[feature_index]
            feature_dtype = infer_features_dtype(dtype, feature_name)

            if features_org == FeatureType.TENSOR:
                part = graph.create_transient_by_name(root, feature_name, feature_dtype, [None])
            else:
                part = graph.create_input(root, feature_name, feature_dtype, [None])
            part = ops.flatten()(part)
            inputs[feature_index] = part
            part = ebm.get_bin_index_on_continuous_value(bins)(part)
            part = ebm.get_bin_score_1d(additive_terms)(part)
            parts.append(part)

        elif feature_type in ['nominal', 'ordinal']:
            col_mapping = model_bins[feature_group[0]][0]
            additive_terms = model.term_scores_[feature_index]

            feature_dtype = infer_features_dtype(dtype, feature_name)
            if features_org == FeatureType.TENSOR:
                raise ValueError("tensor-based inputs are not supported with nominal/ordinal features")

            part = graph.create_input(root, feature_name, feature_dtype, [None])
            if feature_dtype == onnx.TensorProto.BOOL:
                # ONNX converts booleans to strings 0/1, not False/True
                col_mapping = {
                    bool_remap[k]: v
                    for k, v in col_mapping.items()
                }
                model_bins[feature_group[0]][0] = col_mapping
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
                    bin_index = -1 if way_count > len(model_bins[i_feature_index]) else way_count - 1
                    bins = [-np.inf, -np.inf] + list(model_bins[i_feature_index][bin_index])
                    input = graph.strip_to_transients(inputs[i_feature_index])
                    i_parts.append(ebm.get_bin_index_on_continuous_value(bins)(input))

                elif i_feature_type in ['nominal', 'ordinal']:
                    col_mapping = model_bins[i_feature_index][0]
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
    if features_org == FeatureType.TENSOR:
        g = graph.merge(tensor_inputs, *parts)
    else:
        g = graph.merge(*parts)

    if type(model) is ExplainableBoostingClassifier:
        class_type = onnx.TensorProto.STRING if model.classes_.dtype.type is np.str_ else onnx.TensorProto.INT64
        classes = model.classes_
        if class_type == onnx.TensorProto.STRING:
            classes = [c.encode("utf-8") for c in classes]

        g, scores_output_name = ebm.compute_class_score(model.intercept_, explain_name)(g)
        g_scores = graph.strip_to_transients(g)
        if len(model.classes_) == 2:  # binary classification
            g = ebm.predict_class(
                classes=classes, class_type=class_type,
                binary=True, prediction_name=prediction_name
            )(g)
            g = graph.add_output(g, g.transients[0].name, class_type, [None])
            if predict_proba is True:
                gp = ebm.predict_proba(binary=True, probabilities_name=probabilities_name)(g_scores)
                g = graph.merge(graph.clear_transients(g), gp)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, len(model.classes_)])
        else:
            g = ebm.predict_class(
                classes=classes, class_type=class_type,
                binary=False, prediction_name=prediction_name
            )(g)
            g = graph.add_output(g, g.transients[0].name, class_type, [None])
            if predict_proba is True:
                gp = ebm.predict_proba(binary=False, probabilities_name=probabilities_name)(g_scores)
                g = graph.merge(graph.clear_transients(g), gp)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, len(model.classes_)])

        if explain is True:
            if len(model.classes_) == 2:
                g = graph.add_output(g, scores_output_name, onnx.TensorProto.FLOAT, [None, len(model.term_names_), 1])
            else:
                g = graph.add_output(g, scores_output_name, onnx.TensorProto.FLOAT, [None, len(model.term_names_), len(model.classes_)])
    elif type(model) is ExplainableBoostingRegressor:
        g, scores_output_name = ebm.compute_class_score(np.array([model.intercept_]), explain_name)(g)
        g = ebm.predict_value(prediction_name)(g)
        g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None])
        g = graph.add_output(g, scores_output_name, onnx.TensorProto.FLOAT, [None, len(model.term_names_), 1])
    else:
        raise NotImplementedError("{} models are not supported".format(type(model)))

    return g


def to_onnx(model, dtype, name="ebm",
            predict_proba=False,
            explain=False,
            target_opset=None,
            prediction_name="prediction",
            probabilities_name="probabilities",
            explain_name="scores",
            context=None,
            ):
    """Converts an EBM model to ONNX.

    The returned model contains one to three output.
    The first output is always the prediction, and is named "prediction".
    If predict_proba is set to True, then another output named "probabilities" is added.
    If explain is set to True, then another output named "scores" is added.

    Args:
        model: The EBM model, trained with interpretml
        dtype: A dict containing the type of each input feature. Types are expressed as strings, the following values are supported: float, double, int, str.
        name: [Optional] The name of the model
        predict_proba: [Optional] For classification models, output prediction probabilities instead of class
        explain: [Optional] Adds an additional output with the score per feature per class
        target_opset: [Optional][Deprecated] The target onnx opset version to use

    Returns:
        An ONNX model.
    """
    if target_opset:
        logging.warning("to_onnx: target_opset argument is deprecated")
    g = to_graph(
        model=model,
        dtype=dtype,
        name=name,
        predict_proba=predict_proba,
        explain=explain,
        prediction_name=prediction_name,
        probabilities_name=probabilities_name,
        explain_name=explain_name,
        context=context,
        )

    model = graph.to_onnx(g, name=name)
    return model
