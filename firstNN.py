import keras
from keras.layers import Dense

model = keras.models.Sequential()
model.add(Dense(units=1, use_bias=False, input_shape=(1,))) # 仅有一个权重
model.add(Dense(units=1, use_bias=False, input_shape=(1,))) # 仅有一个权重
model.add(Dense(units=1, use_bias=False, input_shape=(1,))) # 仅有一个权重
model.compile(loss='mse', optimizer='adam')


import numpy as np
data_input = np.random.normal(size=100000)
data_label = -(data_input)

model.layers[0].get_weights()

model.fit(data_input, data_label)
result = model.predict(np.array([2.5]))

print(result);
