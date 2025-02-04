import warnings 
import os
import matplotlib.pyplot as plt #type: ignore
import numpy as np #type: ignore
import math
import glob
import shutil

warnings.filterwarnings('ignore')

root_dir = "data"
split_dir = "splitData"
split_folders = ["Training","Testing","Validation"]
number_of_images = {}

for dir in os.listdir(root_dir):
    if not dir.startswith("."):
        fol_path = os.path.join(root_dir, dir)
        if os.path.isdir(fol_path):
            number_of_images[dir] = len(os.listdir(fol_path))

#make a folder structure to properly partition the images
def makeFolders(folder, folderStruct):
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    for dir in folderStruct:
        if not os.path.exists(os.path.join(folder, dir)):
            os.mkdir(os.path.join(folder, dir))
        else:
            print(dir + " Folder Already Exists")

def splitData(origin, destination, split, class_name):

    image_files = [f for f in os.listdir(origin) if not f.startswith('.')]
    available_images = len(image_files)
    num_images = max(0, min(math.floor(split*number_of_images[class_name])-7, available_images))

    if num_images <= 0:  # Skip if there are no valid images to move
        print(f"Skipping {class_name} in {destination}: Not enough images to split.")
        return 

    for img in np.random.choice(a = image_files,
                                size = num_images,
                                replace = False):
        O = os.path.join(origin, img)
        D = os.path.join(destination, img)
        shutil.copy(O, D)
        os.remove(O)

makeFolders("splitData", split_folders)
for dir in os.listdir(split_dir):
    dir_path = os.path.join(split_dir, dir)
    makeFolders(dir_path, os.listdir(root_dir))

for split_folder in split_folders:
    for class_name in number_of_images.keys():
            origin = os.path.join(root_dir, class_name)
            destination = os.path.join(split_dir, split_folder, class_name)
            if "Training" in split_folder:
                split = 0.7
            if "Testing" in split_folder:
                split = 0.15
            if "Validation" in split_folder:
                split = 0.15
            splitData(origin, destination, split, class_name)
            
shutil.rmtree(root_dir)
os.rename(split_dir, root_dir)
