import os
import pytest
import numpy as np
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import ebm2onnx
from .utils import infer_model


def train_titanic_binary_classification(interactions):
    df = pd.read_csv(
        os.path.join('asset','titanic_train.csv'),
        #dtype= {
        #    'Age': np.float32,
        #    'Fare': np.float32,
        #    'Pclass': np.float32, # np.int
        #}
    )
    df = df.dropna()
    feature_columns = ['Age', 'Fare', 'Pclass']
    label_column = "Survived"

    y = df[[label_column]]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    x = df[feature_columns]
    #x = x.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc)
    model = ExplainableBoostingClassifier(interactions=interactions)
    model.fit(x_train, y_train)

    return model, x_test, y_test


def test_predict_binary_classification_no_interactions():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=0)
    pred_ebm = model_ebm.predict(x_test)    

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
        }
    )
    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx)


def test_predict_proba_binary_classification_no_interactions():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=0)
    pred_ebm = model_ebm.predict_proba(x_test)
    pred_ebm_local = model_ebm.explain_local(x_test, y_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        predict_proba=True,
        # explain=True,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
        }
    )
    print(x_test.dtypes)
    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
    })
    
    #print("ebm:")
    #print(pred_ebm)
    #print("onnx:")
    #print(pred_onnx[0])
    #print(pred_ebm_local.data(-1))
    #print(pred_onnx[1])
    
    assert np.allclose(pred_ebm, pred_onnx)

    '''
    for row in range(pred_ebm.shape[0]):
        print(pred_ebm_local.data(row))
        print(pred_onnx[1][row, :])
        print(model_ebm.preprocessor_.col_bin_edges_[0])
        print(model_ebm.additive_terms_[0])
        assert pred_ebm[row, 0] == pytest.approx(pred_onnx[0][row, 0])
    '''
