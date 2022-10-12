import os
import innvestigate

import pandas as pd
from tensorflow import feature_column

from keras.feature_column.dense_features import DenseFeatures
from sklearn.model_selection import train_test_split

import keras
from keras import layers
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate

import tensorflow as tf


tf.compat.v1.disable_eager_execution()


feature_num = 41
csv_file = "./dataset/biodeg_train.csv"
base_dir = os.path.dirname(__file__)
dataframe = pd.read_csv(csv_file, header=None)
print(dataframe.head())
cols = []
for i in range(feature_num + 1):
    cols += ['c' + str(i)]
dataframe.columns = cols

print(dataframe.head())
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('c' + str(feature_num))
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

feature_columns = []
cols.remove('c' + str(feature_num))

for header in cols:
    feature_columns.append(feature_column.numeric_column(header))

feature_layer = DenseFeatures(feature_columns)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = keras.Sequential([
    feature_layer,
    layers.Dense(60, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(50, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(30, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(1, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

model.save('qsar_classifier_v1')
# reloaded_model = keras.models.load_model('qsar_classifier_v1')
# reloaded_model.build()
reloaded_model = model

print(reloaded_model.summary())

csv_test_file = 'dataset/biodeg_test.csv'
bio_test_dataframe = pd.read_csv(csv_test_file, header=None)
cols = []
for i in range(feature_num + 1):
    cols += ['c' + str(i)]
bio_test_dataframe.columns = cols

bio_test_dataframe.pop('c41')
cols.remove('c41')

print(bio_test_dataframe.head())


#---  deep taylor analyzer -------

from innvestigate.backend import graph
model = graph.model_wo_softmax(model)
# Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model)


count = 0
for test_value in bio_test_dataframe.values:
    count += 1
    sample = {cols[i]: test_value[i] for i in range(len(cols))}
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in
                  sample.items()}
    #analyze the specified instance
    a = analyzer.analyze(input_dict)
    print(a)

    predictions = reloaded_model.predict(input_dict,steps=1)
    print(predictions)
    prob = tf.nn.sigmoid(predictions[0],name ='sigmoid')
    # print(tf.compat.v1.Session().run(prob))
    with tf.compat.v1.Session() as sess:
        # print('Input type:', a)
        # print('Input:', sess.run(a))
        # print('Return type:', b)
        print('Output:', sess.run(prob))
    # sess.run(b)
    # print(Tensor.eval(prob))
    if count == 10:
        break







# instance_to_test = data[2][7:8]
#
# model = keras.Sequential([
#     Input(shape=(X_train.shape[1],1)),
#     Dense(units=10, activation="relu", use_bias=False,input_shape=(41,1)),
#     Dense(units=10, activation="relu", use_bias=False),
#     Dense(units=1, activation=activations.softmax, use_bias=False),
#     # layers.Dropout(.1),
#     Dense(1)
# ])
#
# # from innvestigate.backend import graph
# # model = graph.model_wo_softmax(model)
#
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer='adam', metrics=['acc'])
#
# model.fit(X_train,
#           # validation_data=val_ds,
#           y_train,
#           batch_size=5,
#           epochs=2)
#
# # x = tf.ones((1,41))
# # y = model(x)
# from keras.utils import to_categorical
#
# # y_binary = to_categorical(y_train)
#
# print(model.layers[0].get_weights())
# # model.fit(X_train, y_train)
#
# print(model.weights)
# print(model.summary())
# print(y_test[1])
# print(model.predict(y_test[1]))
