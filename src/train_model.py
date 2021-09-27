import json
import os

# Imports
import matplotlib.pyplot as plt

# Tensorflow
from tensorflow import keras
from tensorflow.keras import callbacks, layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Misc imports
import opendatasets as od

# Configure variables for Transfer learning
image_size = 224

target_size = (image_size, image_size)
input_shape = (image_size, image_size, 3)
grid_shape = (1, image_size, image_size, 3)

batch_size = 32


# Define useful functions
def plot_model_history(hist):
    plt.plot(hist["accuracy"], label="accuracy")
    plt.plot(hist["loss"], label="loss")

    if "val_accuracy" in hist and "val_loss" in hist:
        plt.plot(hist["val_accuracy"], label="val_accuracy")
        plt.plot(hist["val_loss"], label="val_loss")

    # Add the labels and legend
    plt.ylabel("Accuracy / Loss")
    plt.xlabel("Epochs #")
    plt.legend()

    # Finally show the plot
    plt.show()


# Download dataset
od.download("https://www.kaggle.com/vipoooool/new-plant-diseases-dataset")

# Join paths
dataset_root = "./new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

train_dir = os.path.join(dataset_root, "train")
test_dir = os.path.join(dataset_root, "valid")

# Define augmentations for train dataset and read the images
train_aug = ImageDataGenerator(
    # Rescale
    rescale=1/255.0,
    # Filling for W/H shift
    fill_mode="nearest",
    # Width and Height shift
    width_shift_range=0.2,
    height_shift_range=0.2,
    # Random zooms
    zoom_range=0.2,
    # Random Shearing aug
    shear_range=0.2,
)

# Read the data from directory
train_data = train_aug.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical"
)

# Augmentations for test data
test_aug = ImageDataGenerator(
    # Rescale
    rescale=1/255.0
)

# Read the data from directory
test_data = test_aug.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical"
)

# Get the list of categories in training data
cats = list(train_data.class_indices.keys())

# Load the base model for transfer learning
mbnet_v2 = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=input_shape
)

# Stop from being trainable
mbnet_v2.trainable = False

# Define the layers
inputs = keras.Input(shape=input_shape)

# Get the layer
x = mbnet_v2(inputs, training=False)

# Stack layers further
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(len(cats), activation="softmax")(x)

# Combine the model
model = Model(inputs=inputs, outputs=x)

# Summary
model.summary()

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Define callbacks to use
early_stopping_cb = callbacks.EarlyStopping(monitor="loss", patience=3)

# Num epochs
epochs = 30

# Train model
print("Starting training loop")
history = model.fit(
    train_data,
    epochs=epochs,
    steps_per_epoch=150,
    callbacks=[early_stopping_cb]
)

print("Training complete. Testing the model")
model.evaluate(test_data)

# Plotting the history of training loop
plot_model_history(history.history)

# Finally save the model and categories
model.save("plant_disease_detection.h5")

with open("categories.json", "w") as file:
    json.dump(train_data.class_indices, file)

print("Saved the model and categories.")
