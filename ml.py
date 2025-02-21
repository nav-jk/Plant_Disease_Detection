import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, 
                                     RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast)
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight

# 🌟 Data Augmentation (More Variations)
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.3),
    RandomZoom(0.3),
    RandomBrightness(0.2),
    RandomContrast(0.2),
])

# 🌟 Load Training Data & Compute Class Weights
training_set = tf.keras.utils.image_dataset_from_directory(
    '/home/navaneetj/Documents/Plant_Disease/train',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    '/home/navaneetj/Documents/Plant_Disease/valid',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

# Extract class names
class_names = training_set.class_names
print("Class Names:", class_names)

# 🌟 Compute Class Weights
labels_list = []
for _, labels in training_set:
    labels_list.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot to class index

class_weights_dict = compute_class_weight(class_weight="balanced",
                                          classes=np.unique(labels_list),
                                          y=labels_list)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}
print("Computed Class Weights:", class_weights)

# 🌟 Model Architecture
model = Sequential()
model.add(data_augmentation)  # Apply augmentation

# Convolutional Layers with BatchNorm Before Activation
model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=3))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=3))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=3))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=3))
model.add(BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Dropout(0.5))  # Increase dropout to prevent overfitting

# Fully Connected Layers with L2 Regularization
model.add(Flatten())
model.add(Dense(units=1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(units=len(class_names), activation='softmax'))  # Adapt to number of classes

# 🌟 Compile Model with Lower Learning Rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ReduceLROnPlateau Callback
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# 🌟 Train Model with Class Weights
training_accuracy = model.fit(training_set, validation_data=validation_set, 
                              epochs=15, callbacks=[lr_scheduler], class_weight=class_weights)

# Evaluate Model
train_loss, train_acc = model.evaluate(training_set)
val_loss, val_acc = model.evaluate(validation_set)
print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

# Save Model
model.save('/home/navaneetj/Documents/Plant_Disease/train/trained_model.keras')

# Print class names
print("Class Names:", class_names)
