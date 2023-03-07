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


def train_titanic_binary_classification(interactions=0, with_categorical=False):
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
    if with_categorical is False:
        feature_types=['continuous', 'continuous', 'continuous', 'continuous']
        feature_columns = ['Age', 'Fare', 'Pclass', 'Old']
    else:
        feature_types=['continuous', 'continuous', 'nominal', 'continuous', 'nominal']
        feature_columns = ['Age', 'Fare', 'Pclass', 'Old', 'Embarked']
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


def train_bank_churners_multiclass_classification(encode_label=True):
    df = pd.read_csv(
        os.path.join('examples','BankChurners.csv'),
    )
    df = df.dropna()
    feature_types=['continuous', 'continuous', 'nominal', 'continuous']
    feature_columns = ['Customer_Age', 'Dependent_count', 'Education_Level', 'Credit_Limit']
    label_column = "Income_Category"

    y = df[[label_column]]
    if encode_label:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        y_enc = y
    x = df[feature_columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc)
    model = ExplainableBoostingClassifier(interactions=0, feature_types=feature_types)
    model.fit(x_train, y_train)

    return model, x_test, y_test


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("interactions", [0, 2, [(0, 1, 2)], [(0, 1, 2, 3)]])
def test_predict_binary_classification(interactions, explain):
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=interactions)
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        explain=explain,
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

    if explain is True:
        assert len(pred_onnx) == 2
    assert np.allclose(pred_ebm, pred_onnx[0])


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("interactions", [0, 2, [(0, 1, 2)], [(0, 1, 2, 3)]])
def test_predict_proba_binary_classification(interactions, explain):
    model_ebm, x_test, y_test = train_titanic_binary_classification(interactions=interactions)
    pred_ebm = model_ebm.predict_proba(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        predict_proba=True,
        explain=explain,
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

    if explain is True:
        assert len(pred_onnx) == 3        
    assert np.allclose(pred_ebm, pred_onnx[1])


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("interactions", [0, 2, [(0, 1, 2)], [(0, 1, 2, 3)]])
def test_predict_regression_without_interactions(interactions, explain):
    model_ebm, x_test, y_test = train_titanic_regression(interactions=0)
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        explain=explain,
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

    if explain is True:
        assert len(pred_onnx) == 2
    assert np.allclose(pred_ebm, pred_onnx[0])


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("interactions", [0, 2, [(0, 1, 2)], [(0, 1, 2, 3)]])
def test_predict_binary_classification_with_categorical(interactions, explain):
    model_ebm, x_test, y_test = train_titanic_binary_classification(
        interactions=interactions,
        with_categorical=True,
    )
    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        explain=explain,
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

    if explain is True:
        assert len(pred_onnx) == 2
    assert np.allclose(pred_ebm, pred_onnx[0])


@pytest.mark.parametrize("encode_label", [False, True])
def test_predict_multiclass_classification(encode_label):
    model_ebm, x_test, y_test = train_bank_churners_multiclass_classification(encode_label=encode_label)
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

    assert (pred_ebm == pred_onnx[0]).all()


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

    assert len(pred_onnx) == 3
    assert np.allclose(pred_ebm, pred_onnx[1])


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
    assert outputs[0].name == "prediction"
    assert outputs[0].shape == [None]
    assert outputs[0].type == 'tensor(int64)'
    assert outputs[1].name == "scores"
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
    assert len(outputs) == 3
    assert outputs[0].name == "prediction"
    assert outputs[0].shape == [None]
    assert outputs[0].type == 'tensor(int64)'
    assert outputs[1].name == "probabilities"
    assert outputs[1].shape == [None, 2]
    assert outputs[1].type == 'tensor(float)'
    assert outputs[2].name == "scores"
    assert outputs[2].shape == [None, 4, 1]
    assert outputs[2].type == 'tensor(float)'



def test_predict_binary_classification_missing_values():
    model_ebm, x_test, y_test = train_titanic_binary_classification(with_categorical=True)

    # patch data
    x_test.iloc[0, x_test.columns.get_loc('Age')] = np.nan
    x_test.iloc[1, x_test.columns.get_loc('Fare')] = np.nan
    x_test.iloc[2, x_test.columns.get_loc('Embarked')] = np.nan

    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        explain=True,
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

    # score of NaN Age on line 0 must be 0
    assert pred_onnx[1][0][0][0] == 0.0  # index: score,iloc,Age, 0

    # score of NaN Fare on line 1 must be 0
    assert pred_onnx[1][1][1][0] == 0.0  # index: score,iloc,Fare, 0

    # score of NaN Embarked on line 2 must be 0
    assert pred_onnx[1][2][4][0] == 0.0  # index: score,iloc,Embarked, 0


def test_predict_binary_classification_unknown_values():
    model_ebm, x_test, y_test = train_titanic_binary_classification(with_categorical=True)

    # patch data
    x_test.iloc[0, x_test.columns.get_loc('Pclass')] = 5
    x_test.iloc[1, x_test.columns.get_loc('Pclass')] = -2
    x_test.iloc[2, x_test.columns.get_loc('Embarked')] = 'Z'

    pred_ebm = model_ebm.predict(x_test)

    model_onnx = ebm2onnx.to_onnx(
        model_ebm,
        explain=True,
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

    # score of Pclass on line 0 must be 0
    assert pred_onnx[1][0][2][0] == 0.0  # index: score,iloc,Pclass, 0

    # score of Pclass on line 1 must be 0
    assert pred_onnx[1][1][2][0] == 0.0  # index: score,iloc,Pclass, 0

    # score of Embarked on line 2 must be 0
    assert pred_onnx[1][2][4][0] == 0.0  # index: score,iloc,Embarked, 0
