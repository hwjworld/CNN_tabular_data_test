# import package
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
import innvestigate

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from innvestigate.backend import graph

feature_number = 41
epoch_number = 100
# load dataset
csv_file = "dataset/biodeg.csv"
dataframe = pd.read_csv(csv_file, header=None, sep=';')
# print(dataframe.head())
dataset = dataframe.values
cols = ["SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-", "C%", "nCp", "nO", "F03[C-N]", "SdssC",
        "HyWi_B(m)", "LOC", "SM6_L", "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR",
        "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]", "SdO", "TI2_L", "nCrt", "C-026",
        "F02[C-N]", "nHDon", "SpMax_B(m)", "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX", "experimental"]

dataframe.columns = cols

# split into input (X) and output (Y) variables
X = dataset[:, 0:feature_number].astype(float)
Y = dataset[:, feature_number]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

print(dataframe.head())
train, test = train_test_split(dataframe, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
# print(len(val), 'validation examples')
print(len(test), 'test examples')

encoder.fit(train.values[:, feature_number])

Y_train = encoder.transform(train.values[:, feature_number]).astype(np.int8)
X_train = np.array(train.values[:, 0:feature_number]).astype(float)  # [..., np.newaxis]

Y_test = encoder.transform(test.values[:, feature_number]).astype(np.int8)
X_test = np.array(test.values[:, 0:feature_number]).astype(float)  # [..., np.newaxis]

model = Sequential()
model.add(Dense(feature_number, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X_train, Y_train, epochs=epoch_number, batch_size=64)

# compute SHAP values
explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)
print("shap values")
print(shap_values)
shap.summary_plot(shap_values[0], plot_type='bar', feature_names=dataframe.columns)

model = graph.model_wo_softmax(model)
analyzer = innvestigate.create_analyzer("deep_taylor", model)
instances_test = np.array_split(X_test, 1)
analyze_results = analyzer.analyze(instances_test)


def calc_mean_relevance(analyze_results):
    """
    计算mean relevalce
    """
    relevances = {}
    for row_idx in range(len(cols) - 1):
        total_r = 0
        for i in range(len(analyze_results)):
            total_r += analyze_results[i][row_idx]
        mean_r = total_r / len(analyze_results)
        relevances[cols[row_idx]] = {'t': total_r, 'm': mean_r}
    return relevances


def get_value_list(instance):
    """
    获取一个病人的feature序列,计算区间使用
    """
    value_list_dict = {}
    for row_idx in range(len(cols) - 1):
        value_list = []
        for day in instance:
            value_list.append(day[row_idx])
        value_list_dict[cols[row_idx]] = value_list
    return value_list_dict


def calc_value_level(value_range_list, feature, value):
    """
    计算每个value的区间
    """
    max_value = max(value_range_list[feature])
    min_value = min(value_range_list[feature])
    # ((input - min) * 100) / (max - min)
    pctg = (value - min_value) / (max_value - min_value)
    return 'Low' if pctg < 0.2 else 'Medium' if pctg < 0.8 else 'High'


csvdata = {'feature': [],
           'time': [],
           'relevance': [],
           'total_relevance': [],
           'mean_relevance': [],
           'value': [],
           'max_value': [],
           'min_value': [],
           'value_level': [],
           'patient': []}

relevance_sum = calc_mean_relevance(analyze_results)
value_list = get_value_list(X_test)
for instance_idx, instance in enumerate(instances_test[0]):
    one_patient = {'feature': [],
                   'time': [],
                   'relevance': [],
                   'value': [],
                   'max_value': [],
                   'min_value': [],
                   'value_level': []}
    # for result_idx, result in enumerate(analyze_results):
    for row_idx in range(len(cols) - 1):
        # Global csv
        csvdata['feature'].append(cols[row_idx])
        csvdata['time'].append(0)
        csvdata['total_relevance'].append(relevance_sum[cols[row_idx]]['t'])
        csvdata['mean_relevance'].append(relevance_sum[cols[row_idx]]['m'])
        revelance = analyze_results[instance_idx][row_idx]
        csvdata['relevance'].append(revelance)
        value = instance[row_idx]
        level = calc_value_level(value_list, cols[row_idx], value)
        csvdata['value'].append(value)
        csvdata['max_value'].append(max(value_list[cols[row_idx]]))
        csvdata['min_value'].append(min(value_list[cols[row_idx]]))
        csvdata['value_level'].append(level)
        csvdata['patient'].append('p' + str(instance_idx))

        # individual csv
        one_patient['feature'].append(cols[row_idx])
        one_patient['time'].append(0)
        one_patient['relevance'].append(revelance)
        one_patient['value'].append(value)
        one_patient['max_value'].append(max(value_list[cols[row_idx]]))
        one_patient['min_value'].append(min(value_list[cols[row_idx]]))
        one_patient['value_level'].append(level)
        # one_patient['patient'].append('p'+str(instance_idx))
    one_patient_df = pd.DataFrame(one_patient)
    one_patient_df.to_csv(
        'datasets/dtds/{}{}.csv'.format('p', str(instance_idx)),
        index=False)

test_result_df = pd.DataFrame(csvdata)
test_result_df.to_csv('datasets/dtd_result.csv', index=False)
