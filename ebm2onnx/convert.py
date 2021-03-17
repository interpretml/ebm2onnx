from collections import namedtuple
from .utils import get_latest_opset_version
from ebm2onnx import graph
from ebm2onnx import ebm
import ebm2onnx.operators as ops

import numpy as np
import onnx

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


onnx_type_for={
    'float': onnx.TensorProto.FLOAT,
    'double': onnx.TensorProto.DOUBLE,
    'int': onnx.TensorProto.INT64,
}


def infer_features_dtype(dtype, feature_name):
    feature_dtype = onnx.TensorProto.DOUBLE
    if dtype is not None:
        feature_dtype = onnx_type_for[dtype[feature_name]]

    return feature_dtype

def to_onnx(model, name=None,
            target_opset=None,
            predict_proba=False,
            explain=False,
            dtype=None,
            ):
    """
    """
    target_opset = target_opset or get_latest_opset_version()
    root = graph.create_graph()

    class_index=0  # model.classes_    => [0,1]
    inputs = [None for _ in model.feature_names]
    parts = []

    # first compute the score of each feature
    for feature_index in range(len(model.feature_names)):
        feature_name=model.feature_names[feature_index]
        feature_type=model.feature_types[feature_index]
        feature_group=model.feature_groups_[feature_index]

        if feature_type == 'continuous':
            bins = [np.NINF, np.NINF] + list(model.preprocessor_.col_bin_edges_[feature_group[0]])
            additive_terms = model.additive_terms_[feature_index]

            feature_dtype = infer_features_dtype(dtype, feature_name)
            part = graph.create_input(root, feature_name, feature_dtype, [None])
            part = ops.flatten()(part)
            inputs[feature_index] = part
            part = ebm.get_bin_index_on_continuous_value(bins)(part)
            part = ebm.get_bin_score_1d(additive_terms)(part)
            parts.append(part)

        elif feature_type == 'interaction':
            bins_0 = [np.NINF, np.NINF] + list(model.pair_preprocessor_.col_bin_edges_[feature_group[0]])
            bins_1 = [np.NINF, np.NINF] + list(model.pair_preprocessor_.col_bin_edges_[feature_group[1]])
            additive_terms = model.additive_terms_[feature_index]

            part_0 = ebm.get_bin_index_on_continuous_value(bins)(inputs[feature_group[0]])
            part_1 = ebm.get_bin_index_on_continuous_value(bins)(inputs[feature_group[1]])
            part = graph.merge(part_0, part_1)
            part = ebm.get_bin_score_2d(np.array(additive_terms))(part)
            parts.append(part)

        else:
            raise NotImplementedError(f"feature type {feature_type} is not supported")

    # Add all scores, and intercept
    g = graph.merge(*parts)
    g, scores_output_name = ebm.compute_class_score(model.intercept_[class_index])(g)

    # post process
    if type(model) is ExplainableBoostingClassifier:
        if len(model.classes_) == 2: # binary classification
            if predict_proba is False:
                g = ebm.predict_class(binary=True)(g)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.INT64, [None])
            else:
                g = ebm.predict_proba(binary=True)(g)
                g = graph.add_output(g, g.transients[0].name, onnx.TensorProto.FLOAT, [None, len(model.classes_)])
        else:
            raise ValueError(f"multi-class classficiation is not supported")        
    else:
        raise ValueError(f"{type(model)} is not supported")

    if explain is True:
        g = graph.add_output(g, scores_output_name, onnx.TensorProto.FLOAT, [None, len(model.feature_names)])
    #print("graph:")
    #print(g.nodes)
    model = graph.compile(g)
    return model
