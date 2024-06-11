import os
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil


dataset_dir = 'dataset'
train_dir = 'dataset/training_set'
val_dir = 'dataset/validation_set'
categories = ['cats', 'dogs']


os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)


for category in categories:
    category_path = os.path.join(dataset_dir, category)
    images = os.listdir(category_path)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    
    for image in train_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(train_dir, category, image)
        shutil.copyfile(src, dst)
    
    
    for image in val_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(val_dir, category, image)
        shutil.copyfile(src, dst)

train_datagen = ImageDataGenerator(
    rescale=1./255,  
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

model.save('model/cat_dog_classifier.h5', save_format='h5')  # Ensure model is saved in HDF5 format

