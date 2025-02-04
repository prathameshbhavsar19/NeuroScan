import numpy as np #type: ignore
import keras #type: ignore
import matplotlib.pyplot as plt #type: ignore
from keras.layers import Flatten, Dense #type: ignore
from keras.models import Model, load_model #type: ignore
from keras.applications.mobilenet import MobileNet, preprocess_input #type: ignore

base_model = MobileNet(input_shape = (299, 299, 3), include_top = False)

for layer in base_model.layers:
    layer.trainable = False

X = Flatten()(base_model.output)
X = Dense(units = 4, activation = 'softmax')(X)

model = Model(base_model.input, X)

base_model.summary()

model.compile(optimizer = 'rmsprop', loss = keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy']) 
