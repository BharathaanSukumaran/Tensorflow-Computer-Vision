import os
import glob
from posixpath import basename
from sklearn.model_selection import train_test_split
import shutil 
import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf
from PIL import Image
from numba import cuda


def clear_memory():
    device = cuda.get_current_device()
    device.reset()
    
# splitting the training data into training data and validation data
def split_data(datapath,trainpath,valpath,split_size=0.1):
    folders = os.listdir(datapath)

    for folder in folders:
        #Concatenate the datapath with the folder name
        full_path = os.path.join(datapath,folder)
        #This module allows us to look into a folder and downlaod all the files in it with an extension of your choice in this case it is png
        images_path = glob.glob(os.path.join(full_path,'*.png*'))
        #splitting the data according to the proportion specified
        (x_train,x_val) = train_test_split(images_path, test_size=split_size)

        for x in x_train:
            #output the name of each image
            basename = os.path.basename(x)
            #Split the data into a single folder containing two folders for training data and validation data
            path_to_folder = os.path.join(trainpath,folder)

            #Check if that folder exists otherwise create it 
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x,path_to_folder)

        for x in x_train:
            #output the name of each image
            basename = os.path.basename(x)
            #Split the data into a single folder containing two folders for training data and validation data
            path_to_folder = os.path.join(valpath,folder)

            #Check if that folder exists otherwise create it 
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x,path_to_folder)

def display_image(images,labels):
   plt.figure(figsize=(10,10))
   
   for i in range(25):
      #chooses a number between 0 and 60000
      idx = np.random.randint(0,images.shape[0]-1)
      img = images[idx]
      label = labels [idx]

      plt.subplot(5,5,i+1)
      plt.title(str(label))
      plt.imshow(img)
   plt.show()


def order_test_set (imagepath,csvpath):

#Create a dictionary with name of the image as the key and the corresponding label as the value 
    testset = {}

    try:
        with open(csvpath,'r') as csvfile:
            #Go through each row in the reader
            reader = csv.reader(csvfile,delimiter=',')
            # In this loop, i holds the row number and row holds the row value
            for i, row in enumerate(reader):
                # We won't use any info from row 0 which has the headers
                if(i == 0):
                    continue
                #Remove the Test/ from the start of the name and get info such as image name and label
                img_name = row[-1].replace('Test/','')
                label = row[-2]
                #Create a path by concatenating the image path and the label where we create a folder with the label as title with the corresponding images in it 
                path_to_folder = os.path.join(imagepath, label)
                #Check to see if folder exists. otherwise create it
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)


                img_full_path = os.path.join(imagepath,img_name)
                #Move the image from the test folder to the new folder with its corresponding label
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO] : Error reading CSV file')


j=-1
def largest_index(image_list):
    global j
    
    largest_value = image_list[0]
    for i in range(len(image_list)):

        if (image_list[i] > largest_value):
            largest_value = image_list[i]
            j = i

    return j 


