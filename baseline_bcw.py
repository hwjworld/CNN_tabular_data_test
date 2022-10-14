import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score


feature_num = 8
padding = 'same' # valid or same

df_train = pd.read_csv("./dataset/train-breast-cancer-wisconsin2.csv",
                       header=None, dtype=float)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("./dataset/test-breast-cancer-wisconsin2.csv",
                      header=None)

Y = np.array(df_train[feature_num].values).astype(np.int8)
X = np.array(df_train[list(range(feature_num))].values)[..., np.newaxis]

Y_test = np.array(df_test[feature_num].values).astype(np.int8)
X_test = np.array(df_test[list(range(feature_num))].values)[..., np.newaxis]


def get_model():
    nclass = 2
    inp = Input(shape=(feature_num, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu,
                          padding=padding)(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding=padding)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding=padding)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding=padding)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding=padding)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding=padding)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding=padding)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding=padding)(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(feature_num, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(feature_num, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

model = get_model()
file_path = "baseline_cnn_bcw.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=2,
                             save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
# callbacks_list = [checkpoint, redonplat]  # early
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=100, verbose=1, callbacks=callbacks_list,
          validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)