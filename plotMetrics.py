import matplotlib.pyplot as plt #type: ignore
import numpy as np #type: ignore
from keras.models import load_model #type: ignore
from train import HS
from dataPreprocessing import test_data
from keras.utils import load_img, img_to_array #type: ignore

h = HS.history
h.keys()

path = "/Users/prathameshbhavsar/Documents/VSCodeProjects/NeuroScan/data/Testing/pituitary/Tr-pi_0679.jpg"

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'])
         
plt.title("Accuracy vs Validation Accuracy")

plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'])
         
plt.title("Loss vs Validation Loss")

plt.show()

model = load_model("./TL_bestmodel.keras")

acc = model.evaluate(test_data)[1]

print(f"The Model Accuracy is {acc*100}%")

img = load_img(path, target_size = (299, 299))
input_arr = img_to_array(img)/255

plt.imshow(input_arr)
plt.show()

input_arr = np.expand_dims(input_arr, axis = 0)

pred = np.argmax(model.predict(input_arr), axis=-1)[0]

if pred == 0:
    print("Image is a Gioma")

if pred == 1:
    print("Image is a Meningioma")

if pred == 2:
    print("Image is a No Tumor")

if pred == 3:
    print("Image is a Pituitary")

