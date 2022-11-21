# Trained model wrapper module structure

In order to make predictions using the trained ML model, the framework neds to have a Python mini-program capable of doing the actual work.

The name of the module should be stated in the [settings.py](../settings.py) file using the TRAINED_MODEL_MODULE_NAME entry. In the absence of this setting, the framework will locate the first file with .py extension in this directory (whose name isn't **\_\_init\_\_.py** or **wrapper.py**) and use it as the trained model module.

## Required functions

This module should implement the following functions (with mypy type hints!):

**def init() -> None:**

This method should deserialise the pickled model file that was trained by a data scientist using the ML toolkit (e.g. Scikit-learn, LightGBM, TensorFlow, Keras), plus any other files needed to transform the input data and return a prediction. This method is called by the framework during the initialisation phase.

**def run(input_data: Iterable) -> Iterable:**

This function does all the ML prediction heavy lifting. It will receive either a Python Iterable (most likely a dictionary) or a Pandas DataFrame as input, do all the necessary transformations, call the trained model's predict() method and return the result.

**def sample() -> Dict:**

This method should return a Python dictionary with sample values for each of the parameters required by the model. This is used in two contexts:

1. Initialisation of the Swagger documentation, listing the JSON parameters and their sample values
2. Validation of the JSON input when the **model/predict** method is invoked

The eagled-eyed reader will have noticed that the methods init() and run() are the same two that Microsoft prescribe when deploying a trained ML model to [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-deploy-and-where). We just added  some type hints, plus sample() to support more advanced functionality:

* OpenAPI documentation
* JSON input validation

An example file is provided: [sample_model.py](sample_model.py)

## Working example

The module [adult_census_income.py](adult_census_income.py) contains a working example of wrapper module for a classifier trained on the ["adult census income"](https://www.kaggle.com/datasets/uciml/adult-census-income) dataset. The folder contains examples of [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [LightGM](https://lightgbm.readthedocs.io) classifiers, trained and pickled as shown [here](https://github.com/jgbustos/ml-demos/blob/master/Census%20Income%20Classification.ipynb).
