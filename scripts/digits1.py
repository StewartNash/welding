from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

#mnist = fetch_openml('mnist_784', version=1)
##print(mnist.keys()) # Keys: data, target, feature_names, DESCR, details, categories, url
#X, y = mnist['data'], mnist['target']
##print(X.shape) # Shape (28 x 28 arrays): (70000, 784)
##print(y.shape) # Shape: (70000, )

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int32)
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes=10)
X_full, X_test, y_full, y_test = train_test_split(X, y,
	test_size=0.2, random_state=42,
	stratify=np.argmax(y, axis=1))
X_train, X_validation, y_train, y_validation = train_test_split(X_full, y_full,
	test_size=0.25, random_state=42,
	stratify=np.argmax(y_full, axis=1))






