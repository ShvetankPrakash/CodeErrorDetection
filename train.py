# Train model to detect errors in code

import os
import glob
import sys
import numpy as np
import tensorflow as tf
from readDataset import readDataset
from tensorflow.keras import layers, models
#import setGPU


# Constants for training
DATASET_DIR = "./dataset_orig"
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 15

def getDataset(rgb=False):
   # Read Dataset
   dataset, labels = readDataset(DATASET_DIR, rgb)
   
   # Normalize data
   dataset = dataset / 256.0 

   # Split Dataset (70/30 split)
   seventy       = int(np.floor(dataset.shape[0] * 0.7))
   trainExamples = dataset[: seventy]
   trainLabels   = labels[: seventy]   

   testExamples = dataset[seventy :]
   testLabels   = labels[seventy :]

   # Create TF Dataset objects
   trainDataset = tf.data.Dataset.from_tensor_slices((trainExamples, trainLabels))
   testDataset  = tf.data.Dataset.from_tensor_slices((testExamples, testLabels))

   return trainDataset, testDataset


def get_multi_class_model():
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


def get_binary_model():
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


def get_resnet():
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

   

def train():
   # Obtain TF datasets
   rgb = True
   trainDataset, testDataset = getDataset(rgb) 

   # Shuffle and batch the datasets
   trainDataset = trainDataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
   testDataset = testDataset.batch(BATCH_SIZE)

   # Obtain and train model
   #model = get_multi_class_model() 
   #model = get_binary_model() 
   model = get_resnet() 
 
   checkpoint_filepath = '../best_model_resnet.h5'
   model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_sparse_categorical_accuracy',
      mode='max',
      save_best_only=True) 

   # Multi-Model Compilation
   #model.compile(optimizer=tf.keras.optimizers.RMSprop(),
   #              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
   #              metrics=['sparse_categorical_accuracy'])

   # Binary Model Compilation
   #model.compile(optimizer=tf.keras.optimizers.RMSprop(),
   #              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
   #              metrics=['accuracy'])

   # Resnet Compilation
   base_learning_rate = 0.0001
   model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 metrics=['accuracy'])

   model.fit(
      trainDataset, 
      epochs=EPOCHS,
      validation_data=testDataset,
      callbacks=[model_checkpoint_callback])

   # Evaluate model 
   print("-------------------EVALUATE ON TEST SET--------------------")
   model.evaluate(testDataset)


if __name__ == "__main__":
   train()
