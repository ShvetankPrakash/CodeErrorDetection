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
DATASET_DIR = "./dataset"
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 15

def getDataset():
   # Read Dataset
   dataset, labels = readDataset(DATASET_DIR)

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


def train():
   # Obtain TF datasets
   trainDataset, testDataset = getDataset() 

   # Shuffle and batch the datasets
   trainDataset = trainDataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
   testDataset = testDataset.batch(BATCH_SIZE)

   # Build and train model
   model = models.Sequential()
   model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Flatten())
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(2))
   model.summary()
 
   model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['sparse_categorical_accuracy'])
   
   model.fit(trainDataset, epochs=EPOCHS)

   # Evaluate model 
   print("-------------------EVALUATE ON TEST SET--------------------")
   model.evaluate(testDataset)


if __name__ == "__main__":
   train()
