import os 
import tensorflow as tf
import numpy as np
from tensorflow import keras 
from keras import optimizers
from keras.preprocessing import image


# Directory with our training horse pictures
train_horse_dir = os.path.join(r'C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join(r'C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/horse-or-human/humans')

# Directory with validation horse pictures
validation_horse_dir = os.path.join(r'C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/validation-horse-or-human/horses')

# Directory with validation human pictures
validation_human_dir = os.path.join(r'C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/validation-horse-or-human/humans')

#see what filenames look lik ein the horse and buman directories
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_names = os.listdir(validation_horse_dir)
print(f'VAL SET HORSES: {validation_horse_names[:10]}')

validation_human_names = os.listdir(validation_human_dir)
print(f'VAL SET HUMANS: {validation_human_names[:10]}')


#total number of pictures in each directory
print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print(f'total validation horse images: {len(os.listdir(validation_horse_dir))}')
print(f'total validation human images: {len(os.listdir(validation_human_dir))}')

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# # Parameters for our graph; we'll output images in a 4x4 configuration
# nrows = 4
# ncols = 4

# # Index for iterating over images
# pic_index = 0

# # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)

# pic_index += 8
# next_horse_pix = [os.path.join(train_horse_dir, fname) 
#                 for fname in train_horse_names[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir, fname) 
#                 for fname in train_human_names[pic_index-8:pic_index]]

# for i, img_path in enumerate(next_horse_pix+next_human_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)

#   img = mpimg.imread(img_path)
#   plt.imshow(img)

# plt.show()

#Building the model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x200 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # # The fifth convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer= optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './computer vision/horse-or-human/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 128 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './computer vision/validation-horse-or-human/',  # This is the source directory for validation images
        target_size=(150, 150),  # All images will be resized to 300x300
        batch_size=32,
        # Since you use binary_crossentropy loss, you need binary labels
        class_mode='binary')

history = model.fit(
      train_generator,  
      epochs=15,
      validation_data = validation_generator)

images = os.listdir(r"C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/tmp/images")

print(images)

for i in images:
 print()
 # predicting images
 path = r"C:\Users\User\OneDrive\Documents\VSCode\Python Projects\Machine Learning and Neural Nets/computer vision/tmp/images/" + i
 img = tf.keras.utils.load_img(path, target_size=(150, 150))
 x = tf.keras.utils.img_to_array(img)
 x /= 255
 x = np.expand_dims(x, axis=0)

 images = np.vstack([x])
 classes = model.predict(images, batch_size=10)
 print(classes[0])
 if classes[0]>0.5:
   print(i + " is a human")
 else:
   print(i + " is a horse")

