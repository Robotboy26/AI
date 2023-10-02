import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys
from PIL import Image
import multiprocessing
import PySimpleGUI as sg

weights_path = None

if len(sys.argv) > 1:
    print(sys.argv[1])
    weights_path = sys.argv[1]

Epochs = None
if len(sys.argv) > 2:
    print(sys.argv[2])
    Epochs = int(sys.argv[2])

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def load_weights():
    layout = [
        [sg.Text("Select a file to load weights from:")],
        [sg.Input(key="-WEIGHTS-", enable_events=True), sg.FileBrowse()],
        [sg.Button("Load Weights"), sg.Button("Skip")]
    ]
    window = sg.Window("Load Weights", layout)
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Skip":
            window.close()
            return None
        elif event == "Load Weights":
            weights_path = values["-WEIGHTS-"]
            if os.path.exists(weights_path):
                window.close()
                return weights_path
    
    window.close()

# Load the weights file
if weights_path == None:
    weights_path = load_weights()

# Define the neural network architecture
def create_neural_network():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))  # Reshape for grayscale images
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # Use 'softmax' for multiclass classification
    return model

def load_images_and_labels(directory):
    images = []
    labels = []
    
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(label_dir, filename)
                    image = Image.open(image_path)
                    image = image.resize((28, 28))  # Resize as needed
                    image_array = np.array(image)  # Convert image to numpy array
                    images.append(image_array)
                    labels.append(int(label))  # Convert label to integer
    
    images = np.array(images)  # Convert the list of image arrays to a numpy array
    labels = np.array(labels)  # Convert labels to numpy array
    
    return images, labels

def main():
    # Create the neural network model
    model = create_neural_network()
    if weights_path != None:
        model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Load images from a directory
    directory = "numbersDataset/trainingSet"
    images, labels = load_images_and_labels(directory)
 
    images = np.reshape(images, (*images.shape, 1))

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode='nearest'  # How to fill newly created pixels after augmentation
    )

    # Generate augmented images
    # Assuming you have 'images' and 'labels' arrays
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    batch_size = 2048

    # Apply transformations and prefetching
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    #augmented_generator = datagen.flow(images, labels, batch_size=batch_size)

    # Train the model using the ImageDataGenerator
    #model.fit(augmented_generator, epochs=10, steps_per_epoch=len(images) // batch_size)
    if Epochs != None:
        num_epochs = Epochs
    else:
        num_epochs = 10

    for epoch in range(num_epochs):
        for batch_images, batch_labels in dataset:
            # Perform one training step using the batch
            loss, accuracy = model.train_on_batch(batch_images, batch_labels)
            print(f"Epoch {epoch + 1}, Loss: {loss}, Accuracy: {accuracy}")


    # Save the model's architecture and weights to a file
    model.save("0.keras")

if __name__ == "__main__":
    main()
