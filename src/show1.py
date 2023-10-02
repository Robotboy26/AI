import PySimpleGUI as sg
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import os
import sys
import subprocess
import random
import time

weightsPath = None
if len(sys.argv) > 1:
    weightsPath = sys.argv[1]

# Load the trained model
if weightsPath == None:
    weightsPath = "0.keras"

model = tf.keras.models.load_model(weightsPath)
folderPath = "/home/robot/Downloads/git/minecraftAI/numbersDataset/trainingSet"

layout = [
    [sg.Text("Select a folder containing subfolders:")],
    [sg.Input(key="-FOLDER-", enable_events=True, default_text=folderPath), sg.FolderBrowse()],
    [sg.Image(key="-IMAGE-", size=(500, 500))],
    [sg.Text("Select the number of images to test (in Test Random Images):")],
    [sg.Input(key="-NUM_IMAGES-", enable_events=True, default_text="500")],
    [sg.Text("Select the number of epochs (in TrainModel):")],
    [sg.Input(key="-EPOCHS-", enable_events=True, default_text="15")],
    [sg.Button("Back"), sg.Button("Next"), sg.Button("Predict"), sg.Button("Test Random Images"), sg.Button("TrainModel"), sg.Button("Exit")],
    [sg.Text("Prediction: ", size=(15, 1)), sg.Text("", key="-PREDICTION-")],
    [sg.Text("Accuracy: ", size=(15, 1)), sg.Text("", key="-ACCURACY-")]
]

window = sg.Window("Image Classifier", layout)

subfolder_paths = []
current_index = 0

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((28, 28))  # Resize to match model input shape
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize pixel values
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

def update_image_display(index):
    if 0 <= index < len(subfolder_paths):
        subfolder_path = subfolder_paths[index]
        image_files = [filename for filename in os.listdir(subfolder_path) if filename.endswith(".jpg")]
        if image_files:
            image_path = os.path.join(subfolder_path, random.choice(image_files))
            image = Image.open(image_path)
            image = image.resize((400, 400))  # Resize to display on the window
            imageTK = ImageTk.PhotoImage(image=image)
            window["-IMAGE-"].update(data=imageTK) 

def test_random_images(num_images=500):
    startTime = time.time()
    correct_predictions = 0
    total_predictions = 0

    for _ in range(num_images):
        subfolder_path = random.choice(subfolder_paths)
        image_files = [filename for filename in os.listdir(subfolder_path) if filename.endswith(".jpg")]
        if image_files:
            image_path = os.path.join(subfolder_path, random.choice(image_files))
            predicted_class = predict_image(image_path)
            if int(predicted_class) == int(os.path.basename(subfolder_path)):
                correct_predictions += 1
            total_predictions += 1

    endTime = time.time()
    timeTaken = endTime - startTime
    accuracy = (correct_predictions / total_predictions) * 100
    window["-ACCURACY-"].update(f"{accuracy:.2f}%, took:{timeTaken}s, {timeTaken / num_images}s per image")

started = False
while True:
    event, values = window.read()
    if started == False:
        folder_path = values["-FOLDER-"]
        subfolder_paths = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
        current_index = 0
        update_image_display(current_index)
        started = True

    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "-FOLDER-":
        folder_path = values["-FOLDER-"]
        subfolder_paths = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
        current_index = 0
        update_image_display(current_index)
    elif event == "Predict":
        if subfolder_paths and current_index < len(subfolder_paths):
            subfolder_path = subfolder_paths[current_index]
            image_files = [filename for filename in os.listdir(subfolder_path) if filename.endswith(".jpg")]
            if image_files:
                image_path = os.path.join(subfolder_path, random.choice(image_files))
                predicted_class = predict_image(image_path)
                if int(predicted_class) == int(os.path.basename(subfolder_path)):
                    Correct = True
                else:
                    Correct = False
                window["-PREDICTION-"].update(f"Predicted class: {predicted_class}, Label: {os.path.basename(subfolder_path)}, Correct: {Correct}")
    elif event == "Next":
        current_index = (current_index + 1) % len(subfolder_paths)
        update_image_display(current_index)
        window["-PREDICTION-"].update("")
    elif event == "Back":
        current_index = (current_index - 1) % len(subfolder_paths)
        update_image_display(current_index)
        window["-PREDICTION-"].update("")
    elif event == "Test Random Images":
        num_images = int(values["-NUM_IMAGES-"]) # default is 500
        test_random_images(num_images)
    elif event == "TrainModel":
        epochs = int(values["-EPOCHS-"]) # default is 15
        command = f"python3 test1.py 0.keras {epochs}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)



window.close()
