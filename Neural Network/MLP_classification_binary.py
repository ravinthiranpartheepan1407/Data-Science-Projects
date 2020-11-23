# Multilayer perceptron

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy


# Sequential - simplest type of model, a linear stack of layers

# fix random seed for reproducibility
numpy.random.seed(7)


# load pima indians dataset
# 2 classes
# 8 features
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#Training set
x_train = X
y_train = Y

#easier way to get input dimension
input_shape = X.shape[1]
#or it could be described manually
#input_shape = 8

# create a multilayer perceptron
model = Sequential()
#add layers .add()
#indicate number of neurons in a hidden layer, input dimension, activation function 'relu'
#Dense(12) is a fully-connected layer with 12 hidden neurins
model.add(Dense(12, input_dim=input_shape, activation='relu'))
#indicate number of neurons, activation function 'relu'
model.add(Dense(8, activation='relu'))
#for 2 classes: 1 output neuron, activation function 'sigmoid'
model.add(Dense(1, activation='sigmoid'))
#for multi-class softmax classification: N_classes output neurons
#model.add(Dense(3, activation='softmax'))

#print model configuration
model.summary()

#compile model
#loss function function to minimize: mse
#optimization method: SGD, Stochastic gradient descent optimizer
#metric: classification accuracy
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# Fit the model
#neural network parameters
#epochs - training iterations
model.fit(x_train, y_train, epochs=200, batch_size=5)

# evaluate the model
scores_train = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
y_train_predictions = model.predict(x_train)

#Round and print predictions on test set
y_train_prediction_class = [round(x[0]) for x in y_train_predictions]
#print(y_train_prediction_class)

#build a text report showing the main classification metrics
report_train= classification_report(y_train,y_train_prediction_class)
print(report_train)



