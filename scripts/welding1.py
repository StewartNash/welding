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
    categorical_encoder = OneHotEncoder()
    myset_categorical = categorical_encoder.fit_transform(myset_categorical)
    myset_categorical = myset_categorical.toarray()
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
train_categorical_encoder = OneHotEncoder()
X_train_categorical = train_categorical_encoder.fit_transform(X_train_categorical)
X_train_categorical = X_train_categorical.toarray()

X_test_labels = X_test[OUTPUT_COLUMNS].copy()
X_test = X_test.drop(OUTPUT_COLUMNS, axis=1)
X_test_categorical = X_test_labels[CATEGORICAL_LABELS]
X_test_labels = X_test_labels.drop(CATEGORICAL_LABELS, axis=1)
test_categorical_encoder = OneHotEncoder()
X_test_categorical = test_categorical_encoder.fit_transform(X_test_categorical)
X_test_categorical = X_test_categorical.toarray()

X_validation_labels = X_validation[OUTPUT_COLUMNS].copy()
X_validation = X_validation.drop(OUTPUT_COLUMNS, axis=1)
X_validation_categorical = X_validation_labels[CATEGORICAL_LABELS]
X_validation_labels = X_validation_labels.drop(CATEGORICAL_LABELS, axis=1)
validation_categorical_encoder = OneHotEncoder()
X_validation_categorical = validation_categorical_encoder.fit_transform(X_validation_categorical)
X_validation_categorical = X_validation_categorical.toarray()

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

y_train = y_train_regression
y_test = y_test_regression
y_validation = y_validation_regression

import tensorflow as tf
from tensorflow import keras

INPUT_WIDTH = len(INPUT_COLUMNS)
OUTPUT_WIDTH = len(OUTPUT_COLUMNS)
numerical_output_width = y_train.shape[1]
categorical_output_width = y_train_classification.shape[1]

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(INPUT_WIDTH,)))
model.add(keras.layers.Dense(INPUT_WIDTH * 10, activation="relu"))
model.add(keras.layers.Dense(INPUT_WIDTH * 5, activation="relu"))
model.add(keras.layers.Dense(INPUT_WIDTH * 1, activation="relu"))
#model.add(keras.layers.Dense(numerical_output_width, activation="softmax"))
model.add(keras.layers.Dense(numerical_output_width))
#model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
history = model.fit(X_train,
    y_train,
    epochs=30,
    validation_data=(X_validation, y_validation))
mse_test = model.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

y_prediction = model.predict(X_test)
print(y_prediction[:10])

model_filename = "my_model.h5"
print("Saving " + model_filename)
#model.save(model_filename)

def f(x):
    return model.predict(x)

def newton_step(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)  # y: (1, output_dim)
    J = tape.jacobian(y, x)  # J: (1, output_dim, 1, input_dim)
    
    # Remove batch dimensions
    J = tf.squeeze(J, axis=[0, 2])  # Now J: (output_dim, input_dim)
    y = tf.squeeze(y, axis=0)       # y: (output_dim,)
    
    JT = tf.transpose(J)            # JT: (input_dim, output_dim)
    JTJ = tf.matmul(JT, J)          # JTJ: (input_dim, input_dim)
    JTf = tf.matmul(JT, tf.expand_dims(y, axis=1))  # JTf: (input_dim, 1)
    
    delta = tf.linalg.solve(JTJ, JTf)  # delta: (input_dim, 1)
    return x - tf.transpose(delta)     # Make sure delta is (1, input_dim)

def gauss_newton(x, number_iterations = 20):
    x = tf.Variable(x, dtype=tf.float32)
    for i in range(number_iterations):
        fx = f(x)
        loss = tf.reduce_sum(fx**2)
        print(f"Step {i}, x = {x.numpy()}, loss = {loss.numpy():.6f}")
        x.assign(newton_step(x))
        
    return x.numpy()

x = gauss_newton(X_test[:1])
                
x = gauss_newton(X_test[:1])
x = scaler.inverse_transform(x)
x = x.flatten().tolist()
print(x)
