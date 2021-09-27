import argparse
import os

# Tensorflow
import tensorflow as tf


# Utility functions
def convert_h5_to_tflite(model, fp):
    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    tflite_model_filepath = os.path.splitext(fp)[0] + ".tflite"

    with open(tflite_model_filepath, "wb") as f:
        f.write(tflite_model)

    print(f"Saved the model as {os.path.splitext(fp)[0]}.tflite")


def handle_invalid_path(filepath):
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"{filepath} does not exist")

    return os.path.expanduser(filepath)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Convert a Keras model to TFLite"
    )

    # Arguments
    parser.add_argument(
        "--model",
        type=handle_invalid_path,
        required=True,
        help="Path to the Keras model to convert"
    )

    # Parse arguments
    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model)

    # Convert the model to TFLite
    convert_h5_to_tflite(model, args.model)
