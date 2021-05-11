# Train model to detect errors in code

import os
import glob
import sys
import numpy as np
import tensorflow as tf
from readDataset import readDataset
#import setGPU


# Constants for training
DATASET_DIR = "./dataset"
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
   

def getDataset():
   # Read Dataset
   dataset, labels = readDataset(DATASET_DIR)

   # Split Dataset (70/30 split)
   seventy       = np.floor(dataset.shape[0] * 0.7)
   trainExamples = dataset[:seventy]
   trainLabels   = labels[:seventy]   

   testExamples = dataset[seventy:]
   testLabels   = labels[seventy:]

   # Create TF Dataset objects
   trainDataset = tf.data.Dataset.from_tensor_slices((trainExamples, trainLabels))
   testDataset  = tf.data.Dataset.from_tensor_slices((testExamples, testLabels))

   return trainDataset, testDataset


def train():
   # Obtain TF datasets
   trainDataset, testDataset = getDataset() 

   # Shuffle and batch the datasets
   train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
   test_dataset = test_dataset.batch(BATCH_SIZE)

   # Build and train model
   model = tf.keras.Sequential([
       tf.keras.layers.Flatten(input_shape=(28, 28)),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)
   ])
   
   model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['sparse_categorical_accuracy'])
   
   model.fit(train_dataset, epochs=10)

   # Evaluate model 
   model.evaluate(test_dataset)


if __name__ == "__main__":
   train()
