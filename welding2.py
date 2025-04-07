import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

TOTAL_ROWS = 10000

COLUMNS = [
    'electrode force',
    'electrode contact surface diameter',
    'squeeze time',
    'weld time',
    'hold time',
    'weld current',
    'leak rate',
    'explosive force',
    'leaking',
    'explosion'
]

INPUT_COLUMNS = [
    'electrode force',
    'electrode contact surface diameter',
    'squeeze time',
    'weld time',
    'hold time',
    'weld current'    
]

OUTPUT_COLUMNS = [
    'leak rate',
    'explosive force',
    'leaking',
    'explosion'
]

CATEGORICAL_LABELS = ['leaking', 'explosion']
REGRESSION_LABELS = ['leak rate', 'explosive force']

def generate_coefficients(input_length, output_length):
    mean = 0
    stddev = 5.0
    rows = input_length
    columns = output_length
    return np.random.normal(mean, stddev, size=(rows, columns))

def generate_row(coefficients):
    rows, columns = coefficients.shape
    input_row = np.random.rand(columns)
    output = np.matmul(coefficients, input_row)
    output = output.tolist()
    assert len(output) >= 4
    output[-2] = "leaking" if output[-4] > 0 else "not leaking"
    output[-1] = "explosion" if output[-3] > 0 else "no explosion"
    output = input_row.tolist() + output
    return output

def generate_sets(myset, output_columns, categorical_labels):
    myset_labels = myset[output_columns].copy()
    myset = myset.drop(output_columns, axis=1)
    myset_categorical = myset_labels[categorical_labels]
    myset_labels.drop(categorical_labels, axis=1)
    categorical_encoder = OneHotEncoder(sparse_output=False)
    myset_categorical = categorical_encoder.fit_transform(myset_categorical)
    return myset, myset_labels, myset_categorical, categorical_encoder

my_dataframe = pd.DataFrame(columns=COLUMNS)
coefficients = generate_coefficients(len(INPUT_COLUMNS), len(OUTPUT_COLUMNS))
for i in range(TOTAL_ROWS):
    my_dataframe.loc[len(my_dataframe)] = generate_row(coefficients)
print(my_dataframe.head())

X_train, X_test = train_test_split(my_dataframe, test_size=0.2, random_state=42)
X_train, X_validation = train_test_split(X_train)

X_train_labels = X_train[OUTPUT_COLUMNS].copy()
X_train = X_train.drop(OUTPUT_COLUMNS, axis=1)
X_train_categorical = X_train_labels[CATEGORICAL_LABELS]
X_train_labels = X_train_labels.drop(CATEGORICAL_LABELS, axis=1)
train_categorical_encoder = OneHotEncoder(sparse_output=False)
X_train_categorical = train_categorical_encoder.fit_transform(X_train_categorical)

X_test_labels = X_test[OUTPUT_COLUMNS].copy()
X_test = X_test.drop(OUTPUT_COLUMNS, axis=1)
X_test_categorical = X_test_labels[CATEGORICAL_LABELS]
X_test_labels = X_test_labels.drop(CATEGORICAL_LABELS, axis=1)
test_categorical_encoder = OneHotEncoder(sparse_output=False)
X_test_categorical = test_categorical_encoder.fit_transform(X_test_categorical)

X_validation_labels = X_validation[OUTPUT_COLUMNS].copy()
X_validation = X_validation.drop(OUTPUT_COLUMNS, axis=1)
X_validation_categorical = X_validation_labels[CATEGORICAL_LABELS]
X_validation_labels = X_validation_labels.drop(CATEGORICAL_LABELS, axis=1)
validation_categorical_encoder = OneHotEncoder(sparse_output=False)
X_validation_categorical = validation_categorical_encoder.fit_transform(X_validation_categorical)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_validation = scaler.transform(X_validation)

print(X_train_categorical[:10])
print(train_categorical_encoder.categories_)

y_train_regression = X_train_labels
y_train_classification = X_train_categorical
y_test_regression = X_test_labels
y_test_classification = X_test_categorical
y_validation_regression = X_validation_labels
y_validation_classification = X_validation_categorical

import tensorflow as tf
from tensorflow import keras

INPUT_WIDTH = len(INPUT_COLUMNS)
OUTPUT_WIDTH = len(OUTPUT_COLUMNS)
#numerical_output_width = X_train_labels.shape[1]
#categorical_output_width = X_train_categorical.shape[1]
CLASSIFICATION_LABELS = CATEGORICAL_LABELS

#model = keras.models.Sequential()
#model.add(keras.layers.InputLayer(input_shape=(INPUT_WIDTH,)))
#model.add(keras.layers.Dense(INPUT_WIDTH * 10, activation="relu"))
#model.add(keras.layers.Dense(INPUT_WIDTH * 5, activation="relu"))
#model.add(keras.layers.Dense(INPUT_WIDTH * 1, activation="relu"))
#model.add(keras.layers.Dense(numerical_output_width))
#model.compile(loss="mse", optimizer="adam", metrics=["mse"])
#history = model.fit(X_train,
#    X_train_labels,
#    epochs=30,
#    validation_data=(X_validation, X_validation_labels))
#mse_test = model.evaluate(X_test, X_test_labels)

inputs = keras.Input(shape=(INPUT_WIDTH,), name="input")
x = keras.layers.Dense(INPUT_WIDTH * 10, activation="relu")(inputs)
x = keras.layers.Dense(INPUT_WIDTH * 5, activation="relu")(x)
regression_output = keras.layers.Dense(len(REGRESSION_LABELS), name="regression")(x)
classification_output = keras.layers.Dense(
    len(CLASSIFICATION_LABELS),
    activation="sigmoid",
    name="classification")(x)
model = keras.Model(
    inputs=inputs,
    outputs=[regression_output, classification_output],
    name="multi_output_model")
model.compile(
    loss={"regression": "mse", "classification": "categorical_crossentropy"},
    optimizer="adam",
    metrics={"regression": ["mae"], "classification": ["accuracy"]})
history = model.fit(
    X_train,
    {"regression": y_train_regression, "classification": y_train_classification},
    validation_data=(X_validation, {"regression": y_validation_regression, "classification": y_validation_classification}),
    epochs=30)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

y_predictions_regression, y_predictions_classification = model.predict(X_test)

