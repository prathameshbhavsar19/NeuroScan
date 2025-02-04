import keras #type: ignore
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D # type: ignore
from keras.models import Sequential # type: ignore
import keras_tuner  as kt #type: ignore


model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape = (299,299,3)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate = 0.3))

model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 4, activation = 'softmax'))

model.summary()
