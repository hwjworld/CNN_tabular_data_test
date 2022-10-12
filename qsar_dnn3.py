from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import innvestigate

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# load dataset
dataframe = read_csv("dataset/biodeg.csv", header=None, sep=';')
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:41].astype(float)
Y = dataset[:, 41]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(41, input_shape=(41,), activation='relu'))
    model.add(Dense(60, input_shape=(41,), activation='relu'))
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    model.add(Dense(1, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


# smaller model
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(20, input_shape=(41,), activation='relu'))
    model.add(Dense(20, input_shape=(20,), activation='relu'))
    model.add(Dense(20, input_shape=(20,), activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


model = create_baseline()


# from innvestigate.backend import graph
# model = graph.model_wo_softmax(model)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=model, epochs=10,
                                          batch_size=5, verbose=2)))

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=2, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)


print("Standardized: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# file_path = "qsar_dnn3_model.h5"
# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=2,
#                              save_best_only=True, mode='max')

# analyzer = innvestigate.create_analyzer("deep_taylor", model)
# a = analyzer.analyze(X)
# print(a)
#
