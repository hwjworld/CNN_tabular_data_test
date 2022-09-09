import pandas as pd

X_train = pd.read_csv(r'dataset/train_6xor_64dim.csv', header=None)
print(X_train.head())

from keras.utils import np_utils
from keras.initializers import he_normal
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

X_train = X_train.replace([-1],0)

print(X_train.describe())
print(X_train.shape)
distrb = X_train.iloc[:, 64].value_counts()
import matplotlib.pyplot as plt

distrb.plot(kind='bar')

import numpy as np

print(X_train.isnull().values.any())
Y_train = X_train[[64]]
X_train.drop([64], axis=1, inplace=True)
print(X_train.shape)

# import pandas as pd
X_test = pd.read_csv(r'dataset/test_6xor_64dim.csv', header=None)
X_test.head()
X_test = X_test.replace([-1],0)

# import numpy as np
print(X_test.isnull().values.any())

Y_test = X_test[[64]]
X_test.drop([64], axis=1, inplace=True)
print(X_test.shape)

y_train = np_utils.to_categorical(Y_train, 2)
y_test = np_utils.to_categorical(Y_test, 2)

nepoch = 30
outlayer = 2
batch_size = 10000

from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers import Dropout
# from keras.layers.merge import concatenate
from keras.layers import concatenate
from keras.utils import plot_model
from keras.layers import Input
from keras.models import Model

input_layer = Input(shape=(64,))

out1 = Dense(64, activation='relu')(input_layer)
out1 = Dropout(0.5)(out1)
out1 = BatchNormalization()(out1)

out2 = Dense(64, activation='relu')(input_layer)
out2 = Dropout(0.5)(out2)
out2 = BatchNormalization()(out2)

out3 = Dense(64, activation='relu')(input_layer)
out3 = Dropout(0.5)(out3)
out3 = BatchNormalization()(out3)

merge = concatenate([out1, out2, out3])

output = Dense(2, activation='sigmoid')(merge)

model = Model(inputs=input_layer, outputs=output)
# summarize layers
print(model.summary())

# plot graph
plot_model(model, to_file='MODEL.png')

adam = optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=nepoch, batch_size=batch_size,
                 validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


def plt_dynamic(x, vy, ty):
    plt.figure(figsize=(10, 5))
    plt.plot(x, vy, 'b', label="Validation Loss")
    plt.plot(x, ty, 'r', label="Train Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy Loss')
    plt.title('\nBinary Crossentropy Loss VS Epochs')
    plt.legend()
    plt.grid()
    plt.show()


import matplotlib.pyplot as plt

x = list(range(1, 31))
vy = hist.history['val_loss']
ty = hist.history['loss']
plt_dynamic(x, vy, ty)
