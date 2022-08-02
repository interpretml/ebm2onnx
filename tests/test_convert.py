import os
import pytest
import numpy as np
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import onnx
import ebm2onnx
from .utils import infer_model, create_session


def train_titanic_binary_classification(interactions, with_categorical=False):
    df = pd.read_csv(
        os.path.join('examples','titanic_train.csv'),
        #dtype= {
        #    'Age': np.float32,
        #    'Fare': np.float32,
        #    'Pclass': np.float32, # np.int
        #}
    )
    df = df.dropna()
    df['Old'] = df['Age'] > 65
    feature_types=['continuous', 'continuous', 'continuous', 'continuous']
    feature_columns = ['Age', 'Fare', 'Pclass', 'Old']
    if with_categorical is True:
        feature_columns.append('Embarked')
        feature_types.append('categorical')
    label_column = "Survived"

    y = df[[label_column]]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    x = df[feature_columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc)
    model = ExplainableBoostingClassifier(interactions=interactions, feature_types=feature_types)
    model.fit(x_train, y_train)

    return model, x_test, y_test


def train_titanic_regression(interactions):
    df = pd.read_csv(os.path.join('examples','titanic_train.csv'))
    df = df.dropna()
    feature_columns = ['SibSp', 'Fare', 'Pclass']
    label_column = "Age"

    y = df[[label_column]]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    x = df[feature_columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc)
    model = ExplainableBoostingRegressor(interactions=interactions)
    model.fit(x_train, y_train)

    return model, x_test, y_test


def train_bank_churners_multiclass_classification():
    df = pd.read_csv(
        os.path.join('examples','BankChurners.csv'),
    )
    df = df.dropna()
    feature_types=['continuous', 'continuous', 'categorical', 'continuous']
    feature_columns = ['Customer_Age', 'Dependent_count', 'Education_Level', 'Credit_Limit']
    label_column = "Income_Category"

    y = df[[label_column]]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    x = df[feature_columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc)
    model = ExplainableBoostingClassifier(interactions=0, feature_types=feature_types)
    model.fit(x_train, y_train)

    return model, x_test, y_test


def test_predict_binary_classification_without_interactions():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=0)
    pred_ebm = model_ebm.predict(x_test)    

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
        }
    )
    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
        'Old': x_test['Old'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_proba_binary_classification_without_interactions():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=0)
    pred_ebm = model_ebm.predict_proba(x_test)
    #pred_ebm_local = model_ebm.explain_local(x_test, y_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        predict_proba=True,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
        }
    )
    print(x_test.dtypes)
    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
        'Old': x_test['Old'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])

    '''
    for row in range(pred_ebm.shape[0]):
        print(pred_ebm_local.data(row))
        print(pred_onnx[1][row, :])
        print(model_ebm.preprocessor_.col_bin_edges_[0])
        print(model_ebm.additive_terms_[0])
        assert pred_ebm[row, 0] == pytest.approx(pred_onnx[0][row, 0])
    '''


def test_predict_binary_classification_wo_interactions_w_explain():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=0)
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        explain=True,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
        }
    )
    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
        'Old': x_test['Old'].values,
    })

    assert len(pred_onnx) == 2
    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_binary_classification_with_interactions():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=2)
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
        }
    )

    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
        'Old': x_test['Old'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_proba_binary_classification_with_interactions():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=2)
    pred_ebm = model_ebm.predict_proba(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        predict_proba=True,
        # explain=True,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
        }
    )
    print(x_test.dtypes)
    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
        'Old': x_test['Old'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_regression_without_interactions():
    model_ebm, x_test, y_test = train_titanic_regression(interactions=0)
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        dtype={
            'SibSp': 'int',
            'Fare': 'double',
            'Pclass': 'int',
        },
    )
    pred_onnx = infer_model(model_onnx, {
        'SibSp': x_test['SibSp'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_regression_with_interactions():
    model_ebm, x_test, y_test = train_titanic_regression(interactions=2)
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        dtype={
            'SibSp': 'int',
            'Fare': 'double',
            'Pclass': 'int',
        },
    )
    pred_onnx = infer_model(model_onnx, {
        'SibSp': x_test['SibSp'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_binary_classification_with_categorical():
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=2, with_categorical=True)
    pred_ebm = model_ebm.predict(x_test)
    print(model_ebm.feature_names)
    print(model_ebm.feature_groups_)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
            'Embarked': 'str'
        }
    )

    pred_onnx = infer_model(model_onnx, {
        'Age': x_test['Age'].values,
        'Fare': x_test['Fare'].values,
        'Pclass': x_test['Pclass'].values,
        'Old': x_test['Old'].values,
        'Embarked': x_test['Embarked'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_multiclass_classification():
    model_ebm, x_test, y_test = train_bank_churners_multiclass_classification()
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        dtype={
            'Customer_Age': 'int',
            'Dependent_count': 'int',
            'Education_Level': 'str',
            'Credit_Limit': 'double',
        }
    )

    pred_onnx = infer_model(model_onnx, {
        'Customer_Age': x_test['Customer_Age'].values,
        'Dependent_count': x_test['Dependent_count'].values,
        'Education_Level': x_test['Education_Level'].values,
        'Credit_Limit': x_test['Credit_Limit'].values,
    })

    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_proba_multiclass_classification():
    model_ebm, x_test, y_test = train_bank_churners_multiclass_classification()
    pred_ebm = model_ebm.predict_proba(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        predict_proba=True,
        explain=True,
        dtype={
            'Customer_Age': 'int',
            'Dependent_count': 'int',
            'Education_Level': 'str',
            'Credit_Limit': 'double',
        }
    )

    pred_onnx = infer_model(model_onnx, {
        'Customer_Age': x_test['Customer_Age'].values,
        'Dependent_count': x_test['Dependent_count'].values,
        'Education_Level': x_test['Education_Level'].values,
        'Credit_Limit': x_test['Credit_Limit'].values,
    })

    assert len(pred_onnx) == 2
    assert np.allclose(pred_ebm, pred_onnx[0])


def test_predict_w_scores_outputs_def():
    model_ebm, _, _ = train_titanic_binary_classification(interactions=0)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        explain=True,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
        }
    )
    session = create_session(model_onnx)

    outputs = session.get_outputs()
    assert len(outputs) == 2
    assert outputs[0].name == "predict_0"
    assert outputs[0].shape == [None]
    assert outputs[0].type == 'tensor(int64)'
    assert outputs[1].name == "scores_0"
    assert outputs[1].shape == [None, 4, 1]
    assert outputs[1].type == 'tensor(float)'


def test_predict_proba_w_scores_outputs_def():
    model_ebm, _, _ = train_titanic_binary_classification(interactions=0)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        predict_proba=True,
        explain=True,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
            'Old': 'bool',
        }
    )
    session = create_session(model_onnx)

    outputs = session.get_outputs()
    assert len(outputs) == 2
    assert outputs[0].name == "predict_proba_0"
    assert outputs[0].shape == [None, 2]
    assert outputs[0].type == 'tensor(float)'
    assert outputs[1].name == "scores_0"
    assert outputs[1].shape == [None, 4, 1]
    assert outputs[1].type == 'tensor(float)'
