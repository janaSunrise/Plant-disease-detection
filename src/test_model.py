import argparse
import json
import os

# Numpy
import numpy as np

# Tensorflow
from tensorflow import keras


# Define functions
def load_categories(file_path):
    with open(file_path) as file:
        data = json.load(file)

    # Swap keys and values
    data = dict([(value, key) for key, value in data.items()])

    return data


def load_model(model_path):
    model = keras.models.load_model(model_path)

    return model


def get_prediction_info(cats, class_id):
    name, disease = cats[class_id].split("___")
    name, disease = name.replace("_", " "), disease.replace("_", " ")

    return name, disease


def get_prediction(model, categories, img):
    # Preprocess image
    img = img.reshape((1, 224, 224, 3))
    img = img.astype("float32") / 255.0

    # Get prediction
    prediction = model.predict(img)
    class_id = prediction.argmax()

    return get_prediction_info(categories, class_id)


def handle_invalid_path(filepath):
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"{filepath} does not exist")

    return os.path.expanduser(filepath)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Load the model, categories and the image file
    parser.add_argument(
        "--model",
        type=handle_invalid_path,
        required=True,
        help="Path to the model file",
    )
    parser.add_argument(
        "--categories",
        type=handle_invalid_path,
        required=True,
        help="Path to the categories file",
    )
    parser.add_argument(
        "--image",
        type=handle_invalid_path,
        required=True,
        help="Path to the image file",
    )

    args = parser.parse_args()

    # Load categories
    categories = load_categories(args.categories)

    # Load model
    model = load_model(args.model)

    # Load image
    img = np.array(
        keras.preprocessing.image.load_img(
            args.image, target_size=(224, 224)
        )
    )

    # Get prediction
    name, disease = get_prediction(model, categories, img)

    # Print prediction
    print(f"The disease is classified as {disease} for {name}")
