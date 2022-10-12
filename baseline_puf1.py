import pandas as pd
X_train = pd.read_csv(r'../input/train_6xor_64dim.csv',header = None)
# print(X_train.head())

# X_train.describe()
# X_train.shape
distrb = X_train.iloc[:,64].value_counts()
distrb.plot(kind = 'bar')
# X_train.isnull().values.any()
Y_train = X_train[[64]]
X_train.drop([64],axis = 1,inplace = True)

from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.layers import Input
from keras.models import Model

input_layer = Input(shape = (64,))

out1 = Dense(64,activation = 'relu')(input_layer)
out1 = Dropout(0.5)(out1)
out1 = BatchNormalization()(out1)

out2 = Dense(64,activation = 'relu')(input_layer)
out2 = Dropout(0.5)(out2)
out2 = BatchNormalization()(out2)

out3 = Dense(64,activation = 'relu')(input_layer)
out3 = Dropout(0.5)(out3)
out3 = BatchNormalization()(out3)

merge = concatenate([out1,out2,out3])

output = Dense(2,activation = 'sigmoid')(merge)

model = Model(inputs=input_layer, outputs=output)
# summarize layers
print(model.summary())

# plot graph
plot_model(model, to_file='MODEL.png')

adam = optimizers.Adam(lr = 0.001)
model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])