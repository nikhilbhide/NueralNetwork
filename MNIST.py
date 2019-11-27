"""
MNIST dataset. Digits 1 - 9 are recognized by CNN
The training accruacy is around 
This is a veyr rudimentary model but still the accuracy is really high
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
import matplotlib.pyplot as plt
image_index = 17777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 3
plt.imshow(x_train[image_index], cmap='jet',interpolation='nearest')

print("Shape",x_train.shape)
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#initializing CNN
classifier = Sequential()

#step 1 - Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(28,28,1), activation='relu'))

#step 2 -   Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

#step 3 - Flattening
classifier.add(Flatten())

#step 4 - Full connection (Classic ANN)
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=10,activation='softmax'))


classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Fit model on training set
classifier.fit(x=x_train,y=y_train, epochs=5)

#Evaluate model on test data set
classifier.evaluate(x=x_test,y=y_test)

#Predict the random number
image_index = 4445
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = classifier.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())