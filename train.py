# Train model to detect errors in code

import os
import glob
import sys
import numpy as np
import tensorflow as tf
from readDataset import readDataset
from tensorflow.keras import layers, models, losses
from autoencoder import Autoencoder
#import setGPU


# Constants for training
DATASET_DIR = "./dataset"
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 20

def getNormals(dataset, rgb=False):
   if rgb:
      normals = np.zeros((0, 80, 80, 3))
   else:
      normals = np.zeros((0, 80, 80, 1))

   for idx, codeBlock in enumerate(dataset):
      if idx % 2 == 0:
         errorBlock = np.copy(codeBlock)
         normals = np.insert(normals, normals.shape[0], errorBlock, 0)

   return normals


def getAnomalies(dataset, rgb=False):
   if rgb:
      anomalies = np.zeros((0, 80, 80, 3))
   else:
      anomalies = np.zeros((0, 80, 80, 1))

   for idx, codeBlock in enumerate(dataset):
      if idx % 2 == 1:
         errorBlock = np.copy(codeBlock)
         anomalies = np.insert(anomalies, anomalies.shape[0], errorBlock, 0)

   return anomalies


def getDataset(rgb=False, autoencoder=False, normalize=False):
   # Read Dataset
   dataset, labels = readDataset(DATASET_DIR, rgb)
   
   # Normalize data
   if normalize:
      dataset = dataset / 255.0 

   # Split Dataset (70/30 split)
   seventy = int(np.floor(dataset.shape[0] * 0.7))

   if autoencoder:
      normalDataset   = getNormals(dataset[: seventy])
      #anomalyData    = getAnomalies()
      testDataset     = dataset[seventy :]
      
      return normalDataset, testDataset

   trainExamples = dataset[: seventy]
   trainLabels   = labels[: seventy]   

   testExamples = dataset[seventy :]
   testLabels   = labels[seventy :]

   # Create TF Dataset objects
   trainDataset = tf.data.Dataset.from_tensor_slices((trainExamples, trainLabels))
   testDataset  = tf.data.Dataset.from_tensor_slices((testExamples, testLabels))

   return trainDataset, testDataset


def getMultiClassCnn():
   # Build model with two neuron output
   model = models.Sequential()
   model.add(layers.Conv2D(32, (3, 3), input_shape=(80, 80, 1)))
   model.add(layers.BatchNormalization())
   model.add(layers.Activation('relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3)))
   model.add(layers.BatchNormalization())
   model.add(layers.Activation('relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3)))
   model.add(layers.BatchNormalization())
   model.add(layers.Activation('relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Flatten())
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(2))
   model.summary()

   return model 


def getBinaryCnn():
   # Build model with single neuron output
   model = models.Sequential()
   model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Flatten())
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
   model.summary()

   return model


def getResnet():
   # Obtain pretrained ResNet and other layers needed to tweak model for our purposes
   resnet = tf.keras.applications.resnet.ResNet50(include_top=False, input_shape=(80,80,3))
   resnet.trainable = False
   preprocess_input = tf.keras.applications.resnet.preprocess_input
   global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
   prediction_layer = tf.keras.layers.Dense(1)

   # Chain pieces together
   inputs = tf.keras.Input(shape=(80,80, 3))
   x = preprocess_input(inputs)
   x = resnet(x, training=False)
   x = global_average_layer(x)
   x = tf.keras.layers.Dropout(0.2)(x)
   outputs = prediction_layer(x)
   model = tf.keras.Model(inputs, outputs)
   
   return model


def getAutoencoder():
   model = Autoencoder()
   return model 


def train():
   # Obtain TF datasets
   rgb = False
   autoencoder = False
   normalize = False
   trainDataset, testDataset = getDataset(rgb, autoencoder, normalize) 

   # Shuffle and batch the datasets
   if not autoencoder:
      trainDataset = trainDataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
      testDataset = testDataset.batch(BATCH_SIZE)

   # Obtain and train model
   model = getMultiClassCnn() 
   #model = getBinaryCnn() 
   #model = getResnet() 
   #model = getAutoencoder()
 
   checkpoint_filepath = '../best_model.h5'
   model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_loss',
      mode='min',
      save_best_only=True) 

   # Multi-Model CNN Compilation
   model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['sparse_categorical_accuracy'])

   # Binary Model CNN Compilation
   #model.compile(optimizer=tf.keras.optimizers.RMSprop(),
   #              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
   #              metrics=['accuracy'])

   # ResNet Compilation
   #base_learning_rate = 0.0001
   #model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
   #              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
   #              metrics=['accuracy'])
   model.fit(
      trainDataset, 
      epochs=EPOCHS,
      validation_data=testDataset,
      callbacks=[model_checkpoint_callback])

   # Autoencoder Compilation
   #model.compile(optimizer='adam', loss=losses.MeanSquaredError())
   #model.fit(trainDataset, trainDataset,
   #             epochs=EPOCHS,
   #             shuffle=True,
   #             validation_data=(testDataset, testDataset),
   #             callbacks=[model_checkpoint_callback])


   # Evaluate model 
   if not autoencoder:
      print("-------------------EVALUATE ON TEST SET--------------------")
      model.evaluate(testDataset)
   
   return model, testDataset


if __name__ == "__main__":
   train()
