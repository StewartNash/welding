import pandas as pd
import numpy as np
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
    categorical_encoder = OneHotEncoder()
    myset_categorical = categorical_encoder.fit_transform(myset_categorical)
    myset_categorical = myset_categorical.toarray()
    return myset, myset_labels, myset_categorical, categorical_encoder

my_dataframe = pd.DataFrame(columns=COLUMNS)
coefficients = generate_coefficients(len(INPUT_COLUMNS), len(OUTPUT_COLUMNS))
for i in range(TOTAL_ROWS):
    my_dataframe.loc[len(my_dataframe)] = generate_row(coefficients)
print(my_dataframe.head())

welding_train, welding_test = train_test_split(my_dataframe, test_size=0.2, random_state=42)
welding_train, welding_validation = train_test_split(welding_train)

welding_train_labels = welding_train[OUTPUT_COLUMNS].copy()
welding_train = welding_train.drop(OUTPUT_COLUMNS, axis=1)
welding_train_categorical = welding_train_labels[CATEGORICAL_LABELS]
train_categorical_encoder = OneHotEncoder()
welding_train_categorical = train_categorical_encoder.fit_transform(welding_train_categorical)
welding_train_categorical = welding_train_categorical.toarray()

welding_test_labels = welding_test[OUTPUT_COLUMNS].copy()
welding_test = welding_test.drop(OUTPUT_COLUMNS, axis=1)
welding_test_categorical = welding_test_labels[CATEGORICAL_LABELS]
test_categorical_encoder = OneHotEncoder()
welding_test_categorical = test_categorical_encoder.fit_transform(welding_test_categorical)
welding_test_categorical = welding_test_categorical.toarray()

welding_validation_labels = welding_validation[OUTPUT_COLUMNS].copy()
welding_validation = welding_validation.drop(OUTPUT_COLUMNS, axis=1)
welding_validation_categorical = welding_validation_labels[CATEGORICAL_LABELS]
validation_categorical_encoder = OneHotEncoder()
welding_validation_categorical = validation_categorical_encoder.fit_transform(welding_validation_categorical)
welding_validation_categorical = welding_validation_categorical.toarray()

scaler = StandardScaler()
welding_train = scaler.fit_transform(welding_train)
welding_test = scaler.transform(welding_test)
welding_validation = scaler.transform(welding_validation)

print(welding_train_categorical[:10])
print(train_categorical_encoder.categories_)

import tensorflow as tf
from tensorflow import keras

print(tf.__file__)
print(tf.__version__)

INPUT_WIDTH = len(INPUT_COLUMNS)
OUTPUT_WIDTH = len(OUTPUT_COLUMNS)
numerical_output_width = welding_train_labels.shape[1]
categorical_output_width = welding_train_categorical.shape[1]

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(INPUT_WIDTH, 1)))
model.add(keras.layers.Dense(INPUT_WIDTH * 10, activation="relu"))
model.add(keras.layers.Dense(INPUT_WIDTH * 5, activation="relu"))
model.add(keras.layers.Dense(INPUT_WIDTH * 1, activation="relu"))
model.add(keras.layers.Dense(numerical_output_width, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(welding_train,
    welding_train_labels,
    epochs=30,
    validation_data=(welding_validation, welding_validation_labels))
mse_test = model.evaluate(welding_test, welding_test_labels)

