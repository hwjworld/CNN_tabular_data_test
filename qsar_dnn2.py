import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import Tensor

from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers.preprocessing import normalization, string_lookup, \
    integer_lookup, category_encoding

from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

tf.compat.v1.disable_eager_execution()

print(tf.__version__)

batch_size = 32
seed = 42
feature_num = 41
csv_file = 'dataset/biodeg_train.csv'

dataframe = pd.read_csv(csv_file, header=None)
cols = []
for i in range(feature_num + 1):
    cols += ['c' + str(i)]
dataframe.columns = cols

print(dataframe.head())


# 将数据帧拆分为训练集、验证集和测试集
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('c41')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


batch_size = 32
train_ds = df_to_dataset(train, shuffle=False, batch_size=batch_size)

# [(train_features, label_batch)] = train_ds.take(1)
# print('Every feature:', list(train_features.keys()))
# print('0', train_features['c0'])
# print('A batch of targets:', label_batch)


# 数值列

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = normalization.Normalization(axis=None)
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)
    return normalizer


# photo_count_col = train_features['PhotoAmt']
# layer = get_normalization_layer('PhotoAmt', train_ds)
# layer(photo_count_col)
# normalizer = normalization.Normalization(axis=None)
# feature_ds = train_ds.map(lambda x, y: x[name])


# 选择要使用的列
batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


all_inputs = []
encoded_features = []

cols.remove('c41')

# Numeric features.
for header in cols:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)



# 创建、编译并训练模型
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dense(60, activation="relu")(all_features)
x = tf.keras.layers.Dense(20, activation="relu")(all_features)
# x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

# tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

model.fit(train_ds, epochs=10, validation_data=val_ds)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

model.save('qsar_classifier')
reloaded_model = tf.keras.models.load_model('qsar_classifier')



sample = {
    "c0": 5.161,
    "c1": 2.5397,
    "c2": 0,
    "c3": 0,
    "c4": 0,
    "c5": 0,
    "c6": 6,
    "c7": 46.4,
    "c8": 0,
    "c9": 5,
    "c10": 0,
    "c11": -0.525,
    "c12": 3.844,
    "c13": 0.745,
    "c14": 10.717,
    "c15": 13,
    "c16": 1.038,
    "c17": 1.112,
    "c18": 0,
    "c19": 0,
    "c20": 0,
    "c21": 1.222,
    "c22": 2,
    "c23": 0,
    "c24": 0,
    "c25": 0,
    "c26": 2.404,
    "c27": 0,
    "c28": 0,
    "c29": 12.008,
    "c30": 2.924,
    "c31": 0,
    "c32": 4,
    "c33": 0,
    "c34": 4,
    "c35": 3.881,
    "c36": 3.037,
    "c37": 0,
    "c38": 8.952,
    "c39": 0,
    "c40": 0
}

csv_test_file = 'dataset/biodeg_test.csv'
bio_test_dataframe = pd.read_csv(csv_test_file, header=None)
cols = []
for i in range(feature_num + 1):
    cols += ['c' + str(i)]
bio_test_dataframe.columns = cols

bio_test_dataframe.pop('c41')
cols.remove('c41')

print(bio_test_dataframe.head())

count = 0
for test_value in bio_test_dataframe.values:
    count += 1
    sample = {cols[i]: test_value[i] for i in range(len(cols))}
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in
                  sample.items()}
    predictions = reloaded_model.predict(input_dict)
    print(predictions)
    prob = tf.nn.sigmoid(predictions[0])
    print(100*prob)
    # print(Tensor.eval(prob))
    if count == 10:
        break


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = reloaded_model.predict(input_dict)
prob = tf.nn.sigmoid(predictions[0])




print(
    "This particular pet had a %.1f percent probability "
    "of getting adopted." % (100 * prob)
)