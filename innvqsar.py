#这篇使用预先编辑好的分别的train和test文件
import innvestigate
import os

import keras.activations
import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, \
    LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, \
    GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import keras.applications.vgg16 as vgg16
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

feature_num = 41
padding = 'same'  # valid or same

base_dir = os.path.dirname(__file__)

df_train = pd.read_csv("./dataset/biodeg_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("./dataset/biodeg_test.csv", header=None)

y_train = np.array(df_train[feature_num].values).astype(np.int8)
X_train = np.array(df_train[list(range(feature_num))].values)[..., np.newaxis]

y_test = np.array(df_test[feature_num].values).astype(np.int8)
X_test = np.array(df_test[list(range(feature_num))].values)[..., np.newaxis]


# create instance to analyze later on
data = (X_train,
        y_train,
        X_test,
        y_test)

instance_to_test = data[2][2:9]


# instance_to_test = test[7:8]


def get_model():
    nclass = 2
    inp = Input(shape=(feature_num, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu,
                          padding=padding)(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu,
                          padding=padding)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu,
                          padding=padding)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu,
                          padding=padding)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu,
                          padding=padding)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu,
                          padding=padding)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu,
                          padding=padding)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu,
                          padding=padding)(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(feature_num, activation=activations.relu, name="dense_1")(
        img_1)
    dense_1 = Dense(feature_num, activation=activations.relu, name="dense_2")(
        dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax,
                    name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    # model.summary()
    return model


# model = get_model()

# Get model
# model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
# Strip softmax layer
# model = innvestigate.utils.model_wo_softmax(model)
from innvestigate.backend import graph

# model = graph.model_wo_softmax(model)

model = get_model()

file_path = "qsar_model.h5"
# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=2,
#                              save_best_only=True, mode='max')
# early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
# redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3,
# verbose=2)
# # callbacks_list = [checkpoint, redonplat]  # early
# callbacks_list = [checkpoint, early, redonplat]  # early
#
# model.fit(X_train, y_train, epochs=100, verbose=2, callbacks=callbacks_list,
#           validation_split=0.1)

model.load_weights(file_path)

model = graph.model_wo_softmax(model)
print(model.predict(instance_to_test))
# model = graph.model_wo_softmax(model)
#
# Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model)

# analyze the specified instance
a = analyzer.analyze(instance_to_test)
print(a)

# Aggregate along color channels and normalize to [-1, 1]
# a = a.sum(axis=np.argmax(np.asarray(a.shape) == 1))
# a /= np.max(np.abs(a))

# Plot
# plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
# plt.imshow(a)
# plt.axis("off")
# plt.savefig(os.path.join(base_dir, "images",
#                          "readme_example_analysis2.png"))


# import seaborn as sns
# sns.set_theme(style="white")
#
#
# # Load the example mpg dataset
# mpg = sns.load_dataset("mpg")

# Plot miles per gallon against horsepower with other semantics
# sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight",
#             sizes=(40, 400), alpha=.5, palette="muted",
#             height=6, data=mpg)


# library & dataset
# import seaborn as sns
# df = sns.load_dataset('iris')
# sns.pairplot(df)
# import matplotlib.pyplot as plt
#
# # # Basic correlogram
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import pandas as pd
#
# # data
# x=["IEEE", "Elsevier", "Others", "IEEE", "Elsevier", "Others"]
# y=[7, 6, 2, 5, 4, 3]
# z=["conference", "journal", "conference", "journal", "conference", "journal"]
#
# # create pandas dataframe
# data_list = pd.DataFrame(
#     {'x_axis': x,
#      'y_axis': y,
#      'category': z
#     })
#
# # change size of data points
# minsize = min(data_list['y_axis'])
# maxsize = max(data_list['y_axis'])
#
# # scatter plot
# sns.catplot(x="x_axis", y="y_axis", kind="swarm", hue="category",sizes=(minsize*100, maxsize*100), data=data_list)
# plt.grid()
#
#
# plt.show()
