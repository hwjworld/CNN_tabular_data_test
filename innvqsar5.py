# 这个是基于innvqsar2.py,将输出的多个时间序列结果改为 one patient one timestamp
import innvestigate

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

# load dataset
csv_file = "dataset/biodeg.csv"
dataframe = pd.read_csv(csv_file, header=None, sep=';')
# print(dataframe.head())
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:41]
X = X.astype(float)
Y = dataset[:, 41]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

cols = ["SpMax_L",
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

dataframe.columns = cols

print(dataframe.head())
train, test = train_test_split(dataframe, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
# print(len(val), 'validation examples')
print(len(test), 'test examples')

encoder.fit(train.values[:, 41])

Y_train = encoder.transform(train.values[:, 41]).astype(np.int8)
X_train = np.array(train.values[:, 0:41]).astype(float)[..., np.newaxis]

Y_test = encoder.transform(test.values[:, 41]).astype(np.int8)
X_test = np.array(test.values[:, 0:41]).astype(float)[..., np.newaxis]


# instance_to_test = X_test[3:8]


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
    model.summary()
    return model


# model = get_model()

# Get model
# Strip softmax layer
from innvestigate.backend import graph
model = get_model()

file_path = "qsar_model2.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=2,
                             save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3,
                              verbose=2)
# callbacks_list = [checkpoint, redonplat]  # early
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X_train, Y_train, epochs=100, verbose=2, callbacks=callbacks_list,
          validation_split=0.1)
model.load_weights(file_path)

# model = graph.model_wo_softmax(model)
# print(model.predict(instance_to_test))
model = graph.model_wo_softmax(model)
#
# Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model)

# analyze the specified instance
# a = analyzer.analyze(instance_to_test)
# print(a)

def calc_mean_relevance(analyze_results):
    """
    计算mean relevalce
    """
    relevances = {}
    for row_idx in range(len(cols) - 1):
        total_r = 0
        for i in range(len(analyze_results)):
            total_r += analyze_results[i][row_idx][0]
        mean_r = total_r / len(analyze_results)
        relevances[cols[row_idx]] = {'t': total_r, 'm': mean_r}
    # print('----calculated mean relevance----')
    # print(relevances)
    return relevances


def get_value_list(instance):
    """
    获取一个病人的feature序列,计算区间使用
    """
    value_list_dict = {}
    for row_idx in range(len(cols) - 1):
        value_list = []
        for day in instance:
            value_list.append(day[row_idx][0])
        value_list_dict[cols[row_idx]] = value_list
    # print('----value_list-----')
    # print(value_list_dict)
    return value_list_dict


def calc_value_level(value_range_list, feature, value):
    """
    计算每个value的区间
    """
    max_value = max(value_range_list[feature])
    range1 = max_value * 0.05
    range2 = max_value * 0.95
    if value <= range1:
        level = 'Low'
    elif range1 < value <= range2:
        level = 'Medium'
    else:
        level = 'High'
    return level


csvdata = {'feature': [],
           'time': [],
           'relevance': [],
           'total_relevance':[],
           'mean_relevance':[],
           'value': [],
           'value_level': [],
           'patient': []}
instances_test = np.array_split(X_test, len(X_test) // 1)

for instance_idx, instance in enumerate(instances_test):
    analyze_results = analyzer.analyze(instance)
    mean_relevance = calc_mean_relevance(analyze_results)
    one_patient = {'feature': [],
                   'time': [],
                   'relevance': [],
                   'total_relevance':[],
                   'mean_relevance':[],
                   'value': [],
                   'value_level': []}
    value_list = get_value_list(instance)
    relevance_sum = calc_mean_relevance(analyze_results)
    for result_idx, result in enumerate(analyze_results):
        for row_idx in range(len(cols) - 1):
            # Global csv
            csvdata['feature'].append(cols[row_idx])
            csvdata['time'].append(str(result_idx * 3))
            # print('{},{},{}'.format(instance_idx, result_idx, row_idx))
            csvdata['relevance'].append(result[row_idx][0])
            csvdata['total_relevance'].append(relevance_sum[cols[row_idx]]['t'])
            csvdata['mean_relevance'].append(relevance_sum[cols[row_idx]]['m'])
            value = instance[result_idx][row_idx][0]
            level = calc_value_level(value_list,cols[row_idx],value)
            csvdata['value'].append(value)
            csvdata['value_level'].append(level)
            csvdata['patient'].append('p' + str(instance_idx))

            # individual csv
            one_patient['feature'].append(cols[row_idx])
            one_patient['time'].append(str(result_idx * 3))
            # print('{},{},{}'.format(instance_idx, result_idx, row_idx))
            one_patient['relevance'].append(result[row_idx][0])
            one_patient['total_relevance'].append(relevance_sum[cols[row_idx]]['t'])
            one_patient['mean_relevance'].append(relevance_sum[cols[row_idx]]['m'])
            one_patient['value'].append(value)
            one_patient['value_level'].append(level)
            # one_patient['patient'].append('p'+str(instance_idx))
        one_patient_df = pd.DataFrame(one_patient)
        one_patient_df.to_csv(
            'datasets/dtds/{}{}.csv'.format('p', str(instance_idx)),
            index=False)

test_result_df = pd.DataFrame(csvdata)
test_result_df.to_csv('datasets/dtd_result.csv', index=False)
