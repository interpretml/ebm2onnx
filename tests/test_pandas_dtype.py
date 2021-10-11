import os
import pandas as pd
import ebm2onnx


def test_titanic_types():
    df = pd.read_csv(os.path.join('examples','titanic_train.csv'))

    df['Survived'] = df['Survived'].astype(bool)
    df['Fare'] = df['Fare'].astype('float32')

    assert ebm2onnx.get_dtype_from_pandas(df) == {
        'PassengerId': 'int',
        'Survived': 'bool',
        'Pclass': 'int',
        'Name': 'str',
        'Sex': 'str',
        'Age': 'double',
        'SibSp': 'int',
        'Parch': 'int',
        'Ticket': 'str',
        'Fare': 'float',
        'Cabin': 'str',
        'Embarked': 'str',
    }
