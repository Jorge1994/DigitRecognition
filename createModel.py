# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 23:04:10 2020

@author: PJ
"""

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load Dataset
mnist_digit_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_digit_dataset.load_data()

# Normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape Data
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)

# Create Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_images, train_labels, epochs=5, batch_size=200, validation_split=0.1)

# Test Model
print("Evaluate Model:")
model.evaluate(test_images, test_labels, verbose=1)

# Save Model  
model.save('digit_model.h5')

