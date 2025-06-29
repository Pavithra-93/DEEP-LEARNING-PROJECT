import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# 🟢 NEW: Data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(32,32,3)),

    # Data augmentation applied during training
    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    # 🟢 NEW: Dropout to reduce overfitting
    layers.Dropout(0.5),

    layers.Dense(10)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train, y_train,
    epochs=20,                 # 🟢 NEW: More epochs
    validation_data=(x_test, y_test)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

import matplotlib.pyplot as plt
import numpy as np

# Get predicted logits
predictions = model.predict(x_test)

# Convert logits to probabilities
probabilities = tf.nn.softmax(predictions)

# Get predicted classes
predicted_classes = np.argmax(probabilities, axis=1)

# Class names for CIFAR-10
class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# Show 10 sample images with predicted and true labels
plt.figure(figsize=(15,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], interpolation="bilinear")
    plt.imshow(x_test[i])
    true_label = class_names[int(y_test[i])]
    pred_label = class_names[predicted_classes[i]]
    color = "green" if pred_label == true_label else "red"
    plt.xlabel(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
plt.tight_layout()
plt.show()
