import keras #type: ignore
from keras.optimizers import Adamax #type: ignore
from keras.callbacks import ModelCheckpoint, EarlyStopping #type: ignore
from dataPreprocessing import train_data, val_data
#For Custom Model
#from model import model
#For Transfer Learning
from transferLearningModel import model
import keras_tuner  as kt #type: ignore

#For Custom Model
#model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#For Transfer Learning
model.compile(optimizer = 'rmsprop', loss = keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy']) 

ES = EarlyStopping(monitor = "val_accuracy", min_delta = 0.01, patience = 6, verbose = 1, mode = 'auto') # patience = 6 for Custom Model

MC = ModelCheckpoint(monitor = "val_accuracy", filepath = "./TL_bestmodel.keras", verbose = 1, save_best_only = True, mode = 'auto')

CD = [ES, MC]

HS = model.fit(train_data, steps_per_epoch = 8, epochs = 30, verbose = 1, validation_data = val_data, validation_steps = 16, callbacks = CD)
