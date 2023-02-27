from gc import callbacks
from re import X
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from my_functions import display_image

# Defining the callback function
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') >=0.75):
            print("\n Reached 75% accuracy so cancelling training!")
            self.model.stop_training = True

# Set the callback function
callbacks = myCallback()

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

#Load the training and test split of the Fashion MNISt dataset
(training_images,training_labels),(test_images,test_labels) = fmnist.load_data()

#Normalise the pixel values of the training and test images
training_images = training_images / 255.0
test_images = test_images / 255.0
# #Set number of characters per row when printing
np.set_printoptions(linewidth=320)

display_image(training_images,training_labels)

# #print the label and image
print(f'LABEL:{training_labels[1]}')
print(f'\n IMAGE PIXEL ARRAY:\n{training_images[1]}')

# #Visualise the image
plt.imshow(training_images[1],cmap='Accent')
plt.show()

#Building the classification model
model = tf.keras.models.Sequential([
    
    # Adding the convolution and pooling layers
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    #Adding the processing layers
    tf.keras.layers.Flatten(),tf.keras.layers.Dense(784, activation = tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.summary()

#Keep in mind that the number of neurons in the last layer must correspond to the number of different data types we are testing. IN this case there are 10 different types of clothing.
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
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])

# The above can also be written as model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy)

#Train the model
model.fit(training_images,training_labels,epochs=5)
#Evaluate the model on unseen data
model.evaluate(test_images,test_labels)
# plt.imshow(test_images[120], cmap ='Greys')
# plt.show()
#Exercise 1
# classifications = model.predict(test_images)
# print(classifications[120])
# print(test_labels[200])
# classifications[0] is an array of probabilities for the first test images. The index with the highest probabilty is the test label

#Effect of changing number of filters
# Increasing the filters increase the training time but number of epochs needed to achieve desired accuraccy decreases and overall accuracy increases

#Effect of removing convolution layers
#Final layer: training time increases but accuracy remains the same

#Effect of adding more convolutions
#Training time increases but accuracy decreases, probably because data has been simplified too much