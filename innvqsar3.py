# 这篇在2的基础上，改变cnn到dense,移除一些没用的代码,,使用tf_tabu_tut1.py的机制
# for one patient and one timestamp
import innvestigate

import pandas as pd
import numpy as np
import tensorflow_datasets as tdfs

from keras import optimizers, losses, activations
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from keras.feature_column.dense_features import DenseFeatures

from sklearn.preprocessing import LabelEncoder
from tensorflow import feature_column

import tensorflow as tf
from tensorflow_addons.layers import CRF

print(tf.__version__)

# import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior();
tf.compat.v1.disable_eager_execution()

feature_num = 41
padding = 'same'  # valid or same

# load dataset
csv_file = "dataset/biodeg.csv"
dataframe = pd.read_csv(csv_file, header=None, sep=';')
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:41].astype(float)
Y = dataset[:, 41]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

cols11 = ["SpMax_L",
        "J_Dz(e)",
        "nHM",
        "F01[N-N]",
        "F04[C-N]",
        "NssssC",
        "nCb-",
        "C%",
        "nCp",
        "nO",
        "F03[C-N]",
        "SdssC",
        "HyWi_B(m)",
        "LOC",
        "SM6_L",
        "F03[C-O]",
        "Me",
        "Mi",
        "nN-N",
        "nArNO2",
        "nCRX3",
        "SpPosA_B(p)",
        "nCIR",
        "B01[C-Br]",
        "B03[C-Cl]",
        "N-073",
        "SpMax_A",
        "Psi_i_1d",
        "B04[C-Br]",
        "SdO",
        "TI2_L",
        "nCrt",
        "C-026",
        "F02[C-N]",
        "nHDon",
        "SpMax_B(m)",
        "Psi_i_A",
        "nN",
        "SM6_B(m)",
        "nArCOOR",
        "nX",
        "experimental"]
cols = []
for i in range(feature_num + 1):
    cols.append('c' + str(i))

print(cols)

dataframe.columns = cols

dataframe['c41'] = np.where(dataframe['c41']=='RB', 1, 0)
dataframe = dataframe.astype(float)
# print(dataframe.head())
train, test = train_test_split(dataframe, test_size=0.2)


train, val = train_test_split(train, test_size=0.2)
# print(len(train), 'train examples')
# print(len(val), 'validation examples')
# print(len(test), 'test examples')
#
# train['c41'] = np.where(train['c41']=='RB', 1, 0)
# test['c41'] = np.where(test['c41']=='RB', 1, 0)
# val['c41'] = np.where(val['c41']=='RB', 1, 0)

encoder.fit(train.values[:, 41])

Y_train = encoder.transform(train.values[:, 41]).astype(np.int8)
X_train = np.array(train.values[:, 0:41]).astype(float)[..., np.newaxis]
#
Y_test = encoder.transform(test.values[:, 41]).astype(np.int8)
X_test = np.array(test.values[:, 0:41]).astype(float)#[..., np.newaxis]

# dataframe.pop('c41')

feature_columns = []
cols.remove('c41')
for col_name in cols:
    feature_columns.append(feature_column.numeric_column(col_name))


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('c41')
    # encoder.fit(labels)
    # labels = encoder.transform(train.values[:, 41]).astype(np.int8)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 20
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# for feature_batch, label_batch in train_ds.take(1):
#     print('Every feature:', list(feature_batch.keys()))
#     print('A batch of ages:', feature_batch['Age'])
#     print('A batch of targets:', label_batch)


feature_layer = DenseFeatures(feature_columns)
# batch_size = 32
# train_ds = df_to_dataset(train, batch_size=batch_size)
# val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    Dense(41, activation=activations.softmax),
    Dense(41, activation='relu'),
    Dropout(.1),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


from innvestigate.backend import graph

# model = graph.model_wo_softmax(model)
#
# model = get_model()
#
# file_path = "qsar_model2.h5"
# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=2,
#                              save_best_only=True, mode='max')
# early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
# redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3,
#                               verbose=2)
# # callbacks_list = [checkpoint, redonplat]  # early
# callbacks_list = [checkpoint, early, redonplat]  # early
#
# model.fit(X_train, Y_train, epochs=100, verbose=2, callbacks=callbacks_list,
#           validation_split=0.1)
# model.load_weights(file_path)
#
# model = graph.model_wo_softmax(model)
# instance_to_test = X_test.transpose()
print(model.predict(test_ds.take(1)))
model = graph.model_wo_softmax(model)
# #
# # Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model)
#
# # analyze the specified instance
# ins = np.array([5,5.0476,1,0,0,0,0,11.1,0,3,0,0,2.872,0.722,9.657,0,1.092,1.153,0,0,0,1.125,0,0,0,0,2,0.446,0,18.375,0.8,0,0,0,1,4.712,4.583,0,9.303,0,0]).astype(float)[..., np.newaxis, np.newaxis]

# test_ds.get_single_element()[0]['c0'][...,np.newaxis]


####以下的代码，花了一天的时间，，搞了半天的Tensor##
single_e = test_ds.get_single_element()[0]
for ele in single_e:
    tf.identity(single_e[ele], name=ele)
    single_e[ele] = single_e[ele][...,np.newaxis]


###截止在这###

print(single_e)
analyzer.analyze(single_e)

ds_n = tdfs.as_numpy(test_ds)
for e in ds_n:
    print(analyzer.analyze(e[0]))
# a = analyzer.analyze(tdfs.as_numpy(test_ds.take(1)).data)
# print(a)
#
# def calc_mean_relevance(analyze_results):
#     """
#     计算mean relevalce
#     """
#     relevances = {}
#     for row_idx in range(len(cols) - 1):
#         total_r = 0
#         for i in range(len(analyze_results)):
#             total_r += analyze_results[i][row_idx][0]
#         mean_r = total_r / len(analyze_results)
#         relevances[cols[row_idx]] = {'t': total_r, 'm': mean_r}
#     # print('----calculated mean relevance----')
#     # print(relevances)
#     return relevances
#
#
# def get_value_list(instance):
#     """
#     获取一个病人的feature序列,计算区间使用
#     """
#     value_list_dict = {}
#     for row_idx in range(len(cols) - 1):
#         value_list = []
#         for day in instance:
#             value_list.append(day[row_idx][0])
#         value_list_dict[cols[row_idx]] = value_list
#     # print('----value_list-----')
#     # print(value_list_dict)
#     return value_list_dict
#
#
# def calc_value_level(value_range_list, feature, value):
#     """
#     计算每个value的区间
#     """
#     max_value = max(value_range_list[feature])
#     range1 = max_value * 0.05
#     range2 = max_value * 0.95
#     if value <= range1:
#         level = 'Low'
#     elif range1 < value <= range2:
#         level = 'Medium'
#     else:
#         level = 'High'
#     return level
#
#
# csvdata = {'feature': [],
#            'time': [],
#            'relevance': [],
#            'total_relevance':[],
#            'mean_relevance':[],
#            'value': [],
#            'value_level': [],
#            'patient': []}
# instances_test = np.array_split(X_test, len(X_test) // 1)
#
# for instance_idx, instance in enumerate(instances_test):
#     analyze_results = analyzer.analyze(instance)
#     mean_relevance = calc_mean_relevance(analyze_results)
#     one_patient = {'feature': [],
#                    'time': [],
#                    'relevance': [],
#                    'total_relevance':[],
#                    'mean_relevance':[],
#                    'value': [],
#                    'value_level': []}
#     value_list = get_value_list(instance)
#     relevance_sum = calc_mean_relevance(analyze_results)
#     for result_idx, result in enumerate(analyze_results):
#         for row_idx in range(len(cols) - 1):
#             # Global csv
#             csvdata['feature'].append(cols[row_idx])
#             csvdata['time'].append(str(result_idx * 3))
#             # print('{},{},{}'.format(instance_idx, result_idx, row_idx))
#             csvdata['relevance'].append(result[row_idx][0])
#             csvdata['total_relevance'].append(relevance_sum[cols[row_idx]]['t'])
#             csvdata['mean_relevance'].append(relevance_sum[cols[row_idx]]['m'])
#             value = instance[result_idx][row_idx][0]
#             level = calc_value_level(value_list,cols[row_idx],value)
#             csvdata['value'].append(value)
#             csvdata['value_level'].append(level)
#             csvdata['patient'].append('p' + str(instance_idx))
#
#             # individual csv
#             one_patient['feature'].append(cols[row_idx])
#             one_patient['time'].append(str(result_idx * 3))
#             # print('{},{},{}'.format(instance_idx, result_idx, row_idx))
#             one_patient['relevance'].append(result[row_idx][0])
#             one_patient['total_relevance'].append(relevance_sum[cols[row_idx]]['t'])
#             one_patient['mean_relevance'].append(relevance_sum[cols[row_idx]]['m'])
#             one_patient['value'].append(value)
#             one_patient['value_level'].append(level)
#             # one_patient['patient'].append('p'+str(instance_idx))
#         one_patient_df = pd.DataFrame(one_patient)
#         one_patient_df.to_csv(
#             'datasets/dtds/{}{}.csv'.format('p', str(instance_idx)),
#             index=False)
#
# test_result_df = pd.DataFrame(csvdata)
# test_result_df.to_csv('datasets/dtd_result.csv', index=False)
