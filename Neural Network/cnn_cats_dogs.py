# Vytautas Magnus University 2019

import keras
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

#for reproducibility
np.random.seed(7)

K.set_image_dim_ordering('tf')

#image numbers for testing (1-15; training is performed using 1-12)
n_test0=10
n_test1=15

# dimensions of our images
img_width, img_height = 150, 150

train_data_dir = 'data_cats_dogs/training'
validation_data_dir = 'data_cats_dogs/testing'
nb_train_samples = 10 
nb_validation_samples = 10
epochs = 50
batch_size = 5

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

# first set of CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# third set of CONV => RELU => POOL
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# set of FC => RELU layers
#model.add(Flatten(input_shape=(None,None)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

# classifier
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary() 

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)




# this is the augmentation configuration we will use for testing:
# only rescaling
#https://keras.io/preprocessing/image/
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

    #steps_per_epoch: Integer or None. Total number of steps (batches of samples) 
    #before declaring one epoch finished and starting the next epoch.
    #When training with input tensors such as TensorFlow data tensors, 
    #the default None is equal to the number of samples in your dataset divided by the batch size,
    #or 1 if that cannot be determined.
    
    #validation_steps: Only relevant if steps_per_epoch is specified. 
    #Total number of steps (batches of samples) to validate before stopping.



#---------------------------------------------------------
# Identification of the unknown object
print("\nIdentification of the unknown object")

name='cat'	
for i in range(n_test0,n_test1+1):
	img_name=name+str(i)+'.jpg'
	test_image=image.load_img(validation_data_dir+'/'+name+'s/'+img_name, target_size = (img_width, img_height))
	test_image=image.img_to_array(test_image)
	test_image=np.expand_dims(test_image, axis = 0)
	result=model.predict(test_image)
   
	if result[0] <=0.5:
		print(img_name+'  CNN decision - CAT: well done!')
	else:
		print(img_name+'  CNN decision - DOG: error!')
	

#  Identification of the unknown object

name='dog'
for i in range(n_test0,n_test1+1):
	img_name=name+str(i)+'.jpg'
	test_image=image.load_img(validation_data_dir+'/'+name+'s/'+img_name, target_size = (img_width, img_height))
	test_image=image.img_to_array(test_image)
	test_image=np.expand_dims(test_image, axis = 0)

	result=model.predict(test_image)     
    
	if result[0] <=0.5:
		print(img_name+'  CNN decision - CAT: error!')
	else:
		print(img_name+'  CNN decision - DOG: well done!')
	
