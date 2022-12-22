========
Ebm2onnx
========


.. image:: https://img.shields.io/pypi/v/ebm2onnx.svg
        :target: https://pypi.python.org/pypi/ebm2onnx

.. image:: https://github.com/interpretml/ebm2onnx/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/interpretml/ebm2onnx/actions/workflows/ci.yml
    :alt: CI

.. image:: https://coveralls.io/repos/github/interpretml/ebm2onnx/badge.svg?branch=master
    :target: https://coveralls.io/github/interpretml/ebm2onnx?branch=master
    :alt: Code Coverage

.. image:: https://readthedocs.org/projects/ebm2onnx/badge/?version=latest
    :target: https://ebm2onnx.readthedocs.io/en/latest/?version=latest
    :alt: Documentation Status

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/interpretml/ebm2onnx/master?filepath=examples%2Fconvert.ipynb


Ebm2onnx converts `EBM <https://github.com/interpretml/interpret>`_ models to
ONNX. It allows to run an EBM model on any ONNX compliant runtime.


Features
--------

* Binary classification
* Regression
* Continuous, nominal, and ordinal variables
* N-way interactions
* Multi-class classification (support is still experimental in EBM)
* Expose local explanations

The export of the models is tested against `ONNX Runtime <https://github.com/Microsoft/onnxruntime>`_. 

Get Started
------------

Train an EBM model:

.. code:: python

    # prepare dataset
    df = pd.read_csv('titanic_train.csv')
    df = df.dropna()

    feature_columns = ['Age', 'Fare', 'Pclass', 'Embarked']
    label_column = "Survived"
    y = df[[label_column]]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    x = df[feature_columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc)

    # train an EBM model
    model = ExplainableBoostingClassifier(
        feature_types=['continuous', 'continuous', 'continuous', 'nominal'],
    )
    model.fit(x_train, y_train)


Then you can convert it to ONNX in a single function call:

.. code:: python

    import onnx
    import ebm2onnx

    onnx_model = ebm2onnx.to_onnx(
        model,
        ebm2onnx.get_dtype_from_pandas(x_train),
    )
    onnx.save_model(onnx_model, 'ebm_model.onnx')


If your dataset is not a pandas dataframe, you can provide the features' types
directly:

.. code:: python

    import ebm2onnx

    onnx_model = ebm2onnx.to_onnx(
        model,
        dtype={
            'Age': 'double',
            'Fare': 'double',
            'Pclass': 'int',
        }
    )
    onnx.save_model(onnx_model, 'ebm_model.onnx')


Try it live
-------------

- You can live test the `model conversion <https://mybinder.org/v2/gh/interpretml/ebm2onnx/master?filepath=examples%2Fconvert.ipynb>`_.
- You can live test `local explanations <https://mybinder.org/v2/gh/interpretml/ebm2onnx/master?filepath=examples%2Fexplain_local.ipynb>`_.


Supporting organizations
-------------------------

The following organizations are supporting Ebm2onnx:

- `SoftAtHome <https://www.softathome.com>`_: Main supporter of Ebm2onnx development.
- `InterpretML <https://interpret.ml>`_: Ebm2onnx is hosted under the umbrella of the InterpretML organization.

|img_sah| |img_interpret|

.. |img_sah| image:: https://raw.githubusercontent.com/interpretml/ebm2onnx/master/assets/sah_logo.png
    :target: https://www.softathome.com

.. |img_interpret| image:: https://raw.githubusercontent.com/interpretml/ebm2onnx/master/assets/interpretml-logo.png
    :target: https://interpret.ml
