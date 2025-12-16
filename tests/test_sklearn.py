import os
import pytest
import numpy as np
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn import compose, impute, pipeline, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType, BooleanTensorType

import onnx
import ebm2onnx
from .utils import infer_model, create_session


def train_titanic_pipeline(interactions=0, old_th=65):
    df = pd.read_csv(
        os.path.join('examples','titanic_train.csv'),
        #dtype= {
        #    'Age': np.float32,
        #    'Fare': np.float32,
        #    'Pclass': np.float32, # np.int
        #}
    )
    df = df.dropna()
    df['Old'] = df['Age'] > old_th
    feature_types = ['continuous', 'continuous', 'continuous', 'continuous']
    feature_columns = ['Age', 'Fare', 'Pclass', 'Old']
    #feature_types = ['continuous', 'continuous']
    #feature_columns = ['Age', 'Fare']
    label_column = "Survived"


    y = df[[label_column]]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    x = df[feature_columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc)

    preprocessor = compose.ColumnTransformer(
        transformers=[
            ("age_scaler", preprocessing.StandardScaler(), ["Age"]),
            ("fare_scaler", preprocessing.StandardScaler(), ["Fare"]),
            ('np_array_transform', 'passthrough', ["Pclass", "Old"]),
        ]
    )

    pipe = pipeline.Pipeline([
        ("preprocessor", preprocessor),
        ("ebm", ExplainableBoostingClassifier(interactions=0, feature_types=feature_types))
    ])

    pipe.fit(x_train, y_train)

    return pipe, x_test, y_test


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("interactions", [0, 2, [(0, 1, 2)], [(0, 1, 2, 3)]])
def test_pipeline_binary_classification(interactions, explain):
    pipe, x_test, y_test = train_titanic_pipeline(
        interactions=interactions
    )

    update_registered_converter(
        ExplainableBoostingClassifier,
        "ExplainableBoostingClassifier",
        ebm2onnx.sklearn.ebm_output_shape_calculator,
        ebm2onnx.sklearn.convert_ebm_classifier,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    model_onnx = convert_sklearn(
        pipe,
        "pipeline_ebm",
        [
            ("Age", FloatTensorType([None, 1])),
            ("Fare", FloatTensorType([None, 1])),
            ("Pclass", Int64TensorType([None, 1])),
            ("Old", BooleanTensorType([None, 1])),
        ],
        options={id(pipe): {"zipmap": False}},
    )

    pred_ebm = pipe.predict(x_test.astype(np.float32))

    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values.reshape((-1, 1)).astype(np.float32),
        'Fare': x_test['Fare'].values.reshape((-1, 1)).astype(np.float32),
        'Pclass': x_test['Pclass'].values.reshape((-1, 1)),
        'Old': x_test['Old'].values.reshape((-1, 1)),
    })

    if explain is True:
        assert len(pred_onnx) == 2

    #assert np.allclose(pred_ebm, pred_onnx[0])
