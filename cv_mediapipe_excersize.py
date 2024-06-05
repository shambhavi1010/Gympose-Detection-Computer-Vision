# -*- coding: utf-8 -*-
"""CV-Mediapipe_Excersize.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hBIGVBVCIspZjxC93ZM643xZX15Zqy_x
"""

pip install mediapipe

"""## PULL UPS"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import scipy.io
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

# Define correct pose angles range
CORRECT_POSE_ANGLE_RANGE = {
    'pull_ups': {
        'elbow': (40, 155),
    },
        'squat': {
        'knee': (60, 180)
    },
        'pushup': {
        'elbow': (90, 160)
    }
}

def process_frames(image_folder):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        # Read each image file
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image for pose estimation
        results = pose.process(image_rgb)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.show()

        # You can also save the results here using cv2.imwrite() if you want

# Call the function with your specific image folder path
process_frames('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/pullup/1149')

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

def process_frames_and_calculate_angles(image_folder):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        # Read each image file
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Assuming you want to calculate the angle of the elbow joint
            # You need to replace the landmarks accordingly
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Display the angle
            print(f"The angle at the elbow is: {angle}")

            # Visualize the result
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print(f"Pose landmarks not detected for image {image_name}")

# Call the function with the path to your specific image folder
process_frames_and_calculate_angles('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/pullup/1149')



def label_data(image_folder, exercise_type):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0]]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0]]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]]

            angle = calculate_angle(shoulder, elbow, wrist)
            correct_range = CORRECT_POSE_ANGLE_RANGE[exercise_type]['elbow']

            label = 'correct' if correct_range[0] <= angle <= correct_range[1] else 'incorrect'

            # You can use this label to save to your dataset or use as needed
            print(f"Image: {image_name}, Exercise: {exercise_type}, Elbow Angle: {angle}, Label: {label}")

            # Display the image with the labeled angle
            cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print(f"Pose landmarks not detected for image {image_name}")

# Replace 'bicep_curl' with your specific exercise type
label_data('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/pullup/1149', 'pull_ups')

import os
from os import listdir, makedirs
from os.path import isfile, join, exists, isdir
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

def process_and_label_data(parent_folder, correct_folder, incorrect_folder):
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    for subfolder in subfolders:
        image_files = [f for f in listdir(subfolder) if isfile(join(subfolder, f))]

        for image_name in image_files:
            image_path = join(subfolder, image_name)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Define the landmarks for the pull-up pose
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0]]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0]]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]]

                angle = calculate_angle(shoulder, elbow, wrist)
                correct_range = CORRECT_POSE_ANGLE_RANGE['pull_ups']['elbow']
                label = 'correct' if correct_range[0] <= angle <= correct_range[1] else 'incorrect'

                # Save the image to the corresponding folder
                save_path = join(correct_folder if label == 'correct' else incorrect_folder, image_name)
                cv2.imwrite(save_path, image)
                print(f"Image: {image_name}, Exercise: pull_ups, Elbow Angle: {angle}, Label: {label}")
            else:
                print(f"Pose landmarks not detected for image {image_name}")

# Define correct angle range for pull-ups
CORRECT_POSE_ANGLE_RANGE = {
    'pull_ups': {
        'elbow': (45, 150),  # Define your range for a correct pull-up
    },
}

# Create directories for saving labeled images if they don't exist
correct_folder = '/content/drive/MyDrive/CV_PROJECT/Processed/Pull_Ups/Correct'
incorrect_folder = '/content/drive/MyDrive/CV PROJECT/Processed/Pull_Ups/Incorrect'

if not exists(correct_folder):
    makedirs(correct_folder)
if not exists(incorrect_folder):
    makedirs(incorrect_folder)

# Path to your pull-up frames
pullup_folder = '/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/pullup'

process_and_label_data(pullup_folder, correct_folder, incorrect_folder)

"""## Squats"""



def process_frames(image_folder):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        # Read each image file
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image for pose estimation
        results = pose.process(image_rgb)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.show()

        # You can also save the results here using cv2.imwrite() if you want

# Call the function with your specific image folder path
process_frames('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/squat/1718')

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def process_frames_and_calculate_angles(image_folder):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image at {image_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Hip, Knee, and Ankle for squat
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate the knee angle
            knee_angle = calculate_angle(hip, knee, ankle)

            print(f"The knee angle for image {image_name} is: {knee_angle}")

            # Visualize the result
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Knee Angle: {knee_angle:.2f} degrees")
            plt.show()
        else:
            print(f"Pose landmarks not detected for image {image_name}")

# Path to your specific image folder
process_frames_and_calculate_angles('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/squat/1718')

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def label_data(image_folder, exercise_type):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1],
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0]]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image.shape[0]]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image.shape[0]]

            angle = calculate_angle(hip, knee, ankle)
            correct_range = CORRECT_POSE_ANGLE_RANGE[exercise_type]['knee']

            label = 'correct' if correct_range[0] <= angle <= correct_range[1] else 'incorrect'

            # You can use this label to save to your dataset or use as needed
            print(f"Image: {image_name}, Exercise: {exercise_type}, Knee Angle: {angle}, Label: {label}")

            # Display the image with the labeled angle
            cv2.putText(image, f"Knee Angle: {angle} ({label})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print(f"Pose landmarks not detected for image {image_name}")

# Call the function with the path to your specific image folder
label_data('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/squat/1718', 'squat')

def process_and_label_data(parent_folder, correct_folder, incorrect_folder):
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    for subfolder in subfolders:
        image_files = [f for f in listdir(subfolder) if isfile(join(subfolder, f))]

        for image_name in image_files:
            image_path = join(subfolder, image_name)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Hip, Knee, and Ankle for squats
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1],
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0]]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image.shape[0]]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image.shape[0]]

                angle = calculate_angle(hip, knee, ankle)
                correct_range = CORRECT_POSE_ANGLE_RANGE['squat']['knee']
                label = 'correct' if correct_range[0] <= angle <= correct_range[1] else 'incorrect'

                # Save the image to the corresponding folder
                save_path = join(correct_folder if label == 'correct' else incorrect_folder, image_name)
                cv2.imwrite(save_path, image)
                print(f"Image: {image_name}, Exercise: squats, Knee Angle: {angle}, Label: {label}")
            else:
                print(f"Pose landmarks not detected for image {image_name}")

import os

correct_folder = '/content/drive/MyDrive/CV/Processed/Squats/Correct'
incorrect_folder = '/content/drive/MyDrive/CV/Processed/Squats/Incorrect'

if not os.path.exists(correct_folder):
    os.makedirs(correct_folder)
if not os.path.exists(incorrect_folder):
    os.makedirs(incorrect_folder)

# Path to your squat frames
squat_folder = '/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/squat'

process_and_label_data(squat_folder, correct_folder, incorrect_folder)

"""##PUSHUPS"""

def process_frames(image_folder):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        # Read each image file
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image for pose estimation
        results = pose.process(image_rgb)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.show()

        # You can also save the results here using cv2.imwrite() if you want

# Call the function with your specific image folder path
process_frames('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/pushup/1441')

def process_frames_and_calculate_angles(image_folder):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image at {image_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Shoulders, Elbows, and Wrists for pushups
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the elbow angle, important for pushup evaluation
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            print(f"The elbow angle for image {image_name} is: {elbow_angle}")

            # Visualize the result
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Elbow Angle: {elbow_angle:.2f} degrees")
            plt.show()
        else:
            print(f"Pose landmarks not detected for image {image_name}")

process_frames_and_calculate_angles('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/pushup/1441')

def label_data(image_folder, exercise_type):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    for image_name in image_files:
        image_path = join(image_folder, image_name)
        image = cv2.imread(image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0]]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0]]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]]

            angle = calculate_angle(shoulder, elbow, wrist)
            correct_range = CORRECT_POSE_ANGLE_RANGE[exercise_type]['elbow']

            label = 'correct' if correct_range[0] <= angle <= correct_range[1] else 'incorrect'

            print(f"Image: {image_name}, Exercise: {exercise_type}, Elbow Angle: {angle}, Label: {label}")

            cv2.putText(image, f"Elbow Angle: {angle} ({label})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print(f"Pose landmarks not detected for image {image_name}")

# Call the function with the path to your specific image folder
label_data('/content/drive/MyDrive/CV PROJECT/Penn_Action/frames/pushup/1441', 'pushup')

"""##MODEL 2"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to training data
correct_folder = '/content/drive/MyDrive/CV/Processed/Pull_Ups/Correct'
incorrect_folder = '/content/drive/MyDrive/CV/Processed/Pull_Ups/Inorrect'

# Create a data generator for loading images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Assuming all images are of the same size
img_width, img_height = 224, 224  # You may change this to the size of your processed images

train_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/CV/Processed/Pull_Ups',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',  # 'binary' because we have a binary classification problem
    subset='training')

validation_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/CV/Processed/Pull_Ups',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='validation')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

epochs=10

# Retrieve the history from the training process
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

# Plot training and validation accuracy and loss
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

plt.show()



"""## Model 3

"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# Define paths
base_dir = '/content/drive/MyDrive/CV/Processed/Pull_Ups'
correct_dir = os.path.join(base_dir, 'Correct')
incorrect_dir = os.path.join(base_dir, 'Incorrect')

# Create a directory for the test set if it doesn't exist
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
    os.mkdir(os.path.join(test_dir, 'Correct'))
    os.mkdir(os.path.join(test_dir, 'Incorrect'))

# Get the list of all files in the correct and incorrect directories
correct_files = [os.path.join(correct_dir, f) for f in os.listdir(correct_dir)]
incorrect_files = [os.path.join(incorrect_dir, f) for f in os.listdir(incorrect_dir)]

# Split the files into training+validation sets and test sets
correct_train, correct_test = train_test_split(correct_files, test_size=0.2, random_state=42)
incorrect_train, incorrect_test = train_test_split(incorrect_files, test_size=0.2, random_state=42)

# Function to copy files to a new directory
def copy_files(files, new_dir):
    for f in files:
        base = os.path.basename(f)
        new_file_path = os.path.join(new_dir, base)
        copyfile(f, new_file_path)

# Copy the test files into the test directory
copy_files(correct_test, os.path.join(test_dir, 'Correct'))
copy_files(incorrect_test, os.path.join(test_dir, 'Incorrect'))

# Create the data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the training and validation generators
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Create the test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model configuration
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
batch_size = 32
no_epochs = 25
no_classes = 2
validation_split = 0.2
verbosity = 1

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Corrected to output a single probability
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Start training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=no_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose=verbosity
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.2f}")

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



"""## ANGLE - IMAGE - CNN

"""

import numpy as np
import os
import cv2

# Lists to store data
image_data = []
angle_data = []
labels = []

# Assuming 'calculate_angle' and 'pose' are defined as before
def prepare_data(image_folder, label_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, image_name)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize image to match model input
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Assume labels are stored in a corresponding folder with the same naming convention
        label = cv2.imread(label_path, 0)  # Assuming label images are in grayscale

        if image is not None and label is not None:
            image_data.append(image)
            labels.append(label)

            # Process for pose and angle
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if results.pose_landmarks:
                # Assuming you have defined hip, knee, and ankle landmarks
                hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                angle_data.append([angle])
            else:
                # Handle case where pose is not detected
                angle_data.append([0])  # Example: Use 0 or some other method to indicate missing angle

# Convert lists to numpy arrays for training
image_data = np.array(image_data)
angle_data = np.array(angle_data)
labels = np.array(labels)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

# Define model architecture
def create_model():
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(224, 224, 64)),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Flatten()),
        LSTM(50, return_sequences=True),
        Dropout(0.5),
        LSTM(50),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
    ])

    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = create_model()
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

# Assume X_train and y_train are your features and labels
X_train, y_train = [], []  # Populate these lists with your preprocessed data

# Similarly prepare X_val and y_val for validation
X_val, y_val = [], []

# Model checkpoint to save the best model
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=20, batch_size=32, callbacks=[checkpoint])