import keras #type: ignore
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.applications.mobilenet import preprocess_input # type: ignore


def preprocessingImages(path):
    # For Custom Model
    #image_data = ImageDataGenerator(brightness_range=(0.8, 1.2), rescale = 1/255) 
    # For Transfet Learning
    image_data = ImageDataGenerator(preprocessing_function = preprocess_input) # type: ignore
    image =  image_data.flow_from_directory(directory = path, target_size = (299,299), batch_size = 32, class_mode = 'sparse')
    class_indices = image.class_indices
    print(f"Detected classes: {image.class_indices}")  # Debugging line
    return image


def preprocessingTestImages(path):
    # For Custom Model
    #image_data = ImageDataGenerator(rescale = 1/255) 
    image_data = ImageDataGenerator(preprocessing_function = preprocess_input) # type: ignore
    image =  image_data.flow_from_directory(directory = path, target_size = (299,299), batch_size = 32, class_mode = 'sparse')
    class_indices = image.class_indices
    print(f"Detected classes: {image.class_indices}") # Debugging line
    return image

train_data = preprocessingImages("data/Training")
test_data = preprocessingTestImages("data/Testing")
val_data = preprocessingTestImages("data/Validation")
