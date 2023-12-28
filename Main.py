import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import warnings
warnings.filterwarnings('ignore')


# Ensure TensorFlow Hub models are loaded in compressed format
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# Function to load an image file and preprocess it
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to display an image
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

# Function to convert a tensor to an image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def perform_style_transfer(content_path, style_path):
    # Load the content and style images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Display the content and style images
    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')
    plt.show()

    # Load TensorFlow Hub model for arbitrary image stylization
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Perform style transfer
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert the tensor to an image and save it
    file_name = 'stylized-image.png'
    tensor_to_image(stylized_image).save(file_name)
    print(f"Style transfer completed and saved as {file_name}")

    # Optionally, display the resulting image
    result_image = tensor_to_image(stylized_image)
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()

def main():
    print("Initializing style transfer program...")

    # Initialize a Tkinter root widget
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Prompt to select the content image
    print("Select the content (target) image")
    content_path = filedialog.askopenfilename(
        title='Select Content Image',
        filetypes=[("image files", ".jpg .jpeg .png")]
    )

    # Prompt to select the style image
    print("Select the style (transfer) image")
    style_path = filedialog.askopenfilename(
        title='Select Style Image',
        filetypes=[("image files", ".jpg .jpeg .png")]
    )

    if content_path and style_path:
        try:
            print("Performing style transfer. This may take a moment...")
            perform_style_transfer(content_path, style_path)
            print("Style transfer completed successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()
