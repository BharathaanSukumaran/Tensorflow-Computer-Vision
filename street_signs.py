from gc import callbacks
from operator import length_hint
from re import X
from subprocess import call
from pyparsing import java_style_comment
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator
import os
from my_functions import display_image, largest_index



print(tf.__version__)
# Defining the callback function
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') >=0.95):
            print("\n Reached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# Set the callback function
callbacks = myCallback()

#Load the training data and validation data
training_images = os.listdir("C:\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\traffic_signs\\train")
val_images = os.listdir("C:\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\traffic_signs\\val")
test_images = os.listdir("C:\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\Test")
#Normalise the pixel values of the training and test images
# training_images = training_images/ 255.0
# test_images = test_images/ 255.0
# Set number of characters per row when printing
# np.set_printoptions(linewidth=320)

# Function to display some pictures
# display_image(training_images)
# print("Image specifications: ", training_images.shape)

# print the label and image
# print(f'LABEL:{training_labels[1]}')
# print(f'\n IMAGE PIXEL ARRAY:\n{training_images[1]}')

# Visualise the image
# plt.imshow(training_images[1],cmap='Accent')
# plt.show()

#Building the classification model
model = tf.keras.models.Sequential([
    #Adding the convolution and pooling layers
    tf.keras.layers.Resizing(64,64),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (64,64,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAvgPool2D(),

    #Adding the processing layers
    tf.keras.layers.Flatten(),tf.keras.layers.Dense(128, activation = tf.nn.relu), tf.keras.layers.Dense(43, activation=tf.nn.softmax)])
      # tf.keras.layers.Flatten(),tf.keras.layers.Dense(25, activation = tf.nn.relu),tf.keras.layers.Dense(100, activation = tf.nn.relu),tf.keras.layers.Dense(150, activation = tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
      

#Keep in mind that the number of neurons in the last layer must correspond to the number of different data types we are testing. IN this case there are 10 different types of pictures.
#Sequential: Defines a sequence of layers in a neural network
#Flatten: Converts matrix of pixels into a one dimensional array
#Dense: Adds a layer of neurons
# Each layer of neurons needs an activation function to tell them what to do:

# For now we are using:

# 1) ReLU:

# if x>0:
#     return X
# else:
#     return 0

# 2)Softmax takes a list of values and scales these so the sum of all elements will be equal to one. When applied to model outputs, the scaled values are the
# probability for that class

# #Declare sample inputs and convert to a tensor
# inputs =  np.array([[1.0,3.0,4.0,2.0]])
# inputs = tf.convert_to_tensor(inputs)
# print(f'input to softmax function:{inputs.numpy()}')

# #Feed the inputs to a softmax activation function
# outputs = tf.keras.activations.softmax(inputs)
# print(f'output of softmax function:{outputs.numpy()}')

# #Get the sum of all values after the softmax
# sum = tf.reduce_sum(outputs)
# print(f'sum of outputs: {sum}')

# #Get the index with the highest value
# prediction = np.argmax(outputs)
# print(f'class with highest probability:{prediction}')

#Define the optimiser and loss functions
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics =['accuracy'])

# The above can also be written as model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy)



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './computer vision/traffic_signs/train',  # This is the source directory for training images
        target_size=(64, 64),  # All images will be resized to 64x64
        batch_size=128,
        color_mode='rgb',
        shuffle = True,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode="categorical")


# Flow validation images in batches of 128 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './computer vision/traffic_signs/val',  # This is the source directory for validation images
        target_size=(64, 64),  # All images will be resized to 64x64
        batch_size=128,
        color_mode='rgb',
        shuffle = True,
        # Since you use binary_crossentropy loss, you need binary labels
        class_mode="categorical")


#Train the model
model.fit(train_generator,epochs=20, callbacks = [callbacks], validation_data = validation_generator,validation_steps=8)
#Evaluate the model on unseen data
# model.evaluate(test_images)
# images = os.listdir(r"C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/tmp/images")
# data_path = r'C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/tmp/images'
# for i in images:
#  print()
#  # predicting images
#  path = os.path.join(data_path,i)
#  img = tf.keras.utils.load_img(path, target_size=(32,32))
#  x = tf.keras.utils.img_to_array(img)
#  x /= 255
#  x = np.expand_dims(x, axis=0)

#  images = np.vstack([x])
#  predictions = model.predict(images, batch_size=10)
#  guess = largest_index(predictions[0])
#  if(predictions[0][guess] > 0.5 and guess != -1):
#    if(guess == 0):
#       print(i + " is an airplane")
#    elif(guess == 1):
#       print(i + " is a vehicle")
#    elif(guess == 2):
#       print(i +" is a bird")
#    elif(guess == 3):
#       print(i +" is a cat")
#    elif(guess == 4):
#       print (i +" is a deer")
#    elif(guess == 5):
#       print(i +" is a dog")
#    elif(guess == 6):
#       print(i+ " is a frog")
#    elif(guess == 7):
#       print(i +" is a horse")
#    elif(guess == 8):
#       print(i+ " is a ship")
#    elif(guess == 9):
#       print(i +" is a truck") 

#  else:
#    print("No proper matches found")
 



# plt.imshow(test_images[120], cmap ='Greys')
# plt.show()
#Exercise 1
# classifications = model.predict(test_images)
# print(classifications[120])
# print(test_labels[120])
# classifications[0] is an array of probabilities for the first test images. The index with the highest probabilty is the test label

#Effect of changing number of filters
# Increasing the filters increase the training time but number of epochs needed to achieve desired accuraccy decreases and overall accuracy increases

#Effect of removing convolution layers
#Final layer: training time increases but accuracy remains the same

#Effect of adding more convolutions
#Training time increases but accuracy decreases, probably because data has been simplified too much

