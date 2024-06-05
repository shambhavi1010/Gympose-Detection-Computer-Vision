# Gympose-Detection-Computer-Vision
<div style="border-radius:12px; padding: 20px; background-color: #d5d5ed; font-size:120%; text-align:center">
# Gympose-Detection-Computer-Vision


## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Workflow](#workflow)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Used](#model-used)
6. [Training](#training)
7. [Results](#results)  
8. [Future Work](#future-work)
9. [Conclusion](#conclusion)
# Introduction
In contemporary society, there's an increasing emphasis on health, diet, nutrition, and fitness, with individuals investing significant time and resources to achieve a balanced lifestyle. Environmental risks and unhealthy food drive people to focus on fitness tracking and health monitoring. Computer vision technology enhances gym workouts by providing automated, real-time feedback on exercise form, preventing injuries, and optimizing techniques. By identifying body alignment and motion through pose estimation, it can recognize exercises and assess form accuracy. This system offers personalized fitness guidance, helps users set realistic goals, and provides intuitive data representations. It revolutionizes fitness tracking and allows healthcare professionals to remotely support clients.

# Installation
To run the code in this project, you will need the following dependencies installed:

1.Python: Version 3.6 or higher
2.TensorFlow: Version 2.0 or higher
3.NumPy: For numerical computations
4.Matplotlib: For data visualization
5.Pandas: For data manipulation and preprocessing
# Workflow 
![Flowchart Building (1)]("C:\Users\shamb\Desktop\CV Project Flow.png")



# Dataset Preparation

The Penn Action dataset contains around 2400 short video frames of various physical movements, including running, jumping, and gym exercises like squats and pull-ups. It features samples in different settings, both indoor and outdoor, with diverse individuals. Annotations include key-points like hips, knees, and elbows, captured under varying camera angles and lighting conditions. The 320 x 240 pixel resolution ensures efficient processing and storage while preserving essential details. This dataset is widely used for human pose estimation and action recognition in computer vision tasks.
Since penn action dataset contains a set of 2236 videos with
involving a broad range of 15 actions, but for the purpose of
our model we only needed those which had gym and exercise
related poses in them, which were eight actions out of the 15,
these eight being : pullup, pushup, situp, squat, bench press,
jumping jacks, clean and jerk and jump rope.Bbox was the bounding box in which
the action occurred and dimension indicates the dimension of
the image, using scipy based libraries of python we filtered
out the video files and kept relevant ones belonging to the 8
categories stated above, after all filtering we ended with 1163
videos.


# Model Used
MediaPipe Pose detection involves setting up components and input/output streams for processing image frames uniformly. It extracts key points for pose detection, calculates alignment angles, and visualizes results using input data from image sequences or camera footage. The system uses a pre-trained deep learning model with convolutional neural networks to locate body parts like shoulders, elbows, and wrists. Post-processing refines results and smooths jittering in video sequences to eliminate outliers.


# Training
The author discusses the key components of the Pose model in MediaPipe, using backbone networks like ResNet, MobileNet, or EfficientNet for feature extraction. A pose estimation head with convolutional layers locates key points using confidence scores. Diverse and standardized annotated images train the model for robustness. The author focuses on three gym exercises from the Penn Action dataset: pull-ups, squats, and push-ups, using OpenCV and MediaPipe for image processing. MediaPipe's Pose class detects and visualizes key points, with specific angle ranges for correct form. Functions are defined to process frames, calculate angles, and label images as 'correct' or 'incorrect,' saving labeled images for machine learning model improvement. The pipeline is applied similarly to squats and push-ups, considering relevant key points for each exercise.
### Data Preparation
Before training can begin, your Gym pose action data needs to be formatted correctly. Mediapipe is installed post which these training steps are involved. Taking 'pull-ups' as an instance:
- **Pose drawing using mp_pose**:Pose is drawn using Media Pipe's mp_drawing with arguements passed for detection confidence and static image mode.

### Training Steps
The training process involves several steps that are executed in each epoch:

1. **Correct Pose Angle Range**:
   - For every pose the correct angle range is defined for pull-ups: elbow, squat: knee, pushup: elbow.

2. **Processed Frames**:
   - Model iterates through every image in the folder, conversion of BGR to RGB, landmarks drawn and using matplotlib library results obtained.

3. **Angle Calculation**:
   - Arctan function of numpy used to return the angle made.

4. **Pose Estimation and Angle Calculation**:
   -The code processes images from a specified folder to detect pose landmarks using MediaPipe. It converts images from BGR to RGB, calculates the angle at the elbow joint, and visualizes the pose with landmarks. If landmarks are detected, the elbow angle is printed and displayed on the image.








5. **Incorrect/ Correct Pose Estimation**:
  The code processes images to detect pose landmarks, calculates the elbow angle, and labels each image as 'correct' or 'incorrect' based on predefined angle ranges for a specific exercise type. It converts images to RGB, processes them with MediaPipe, and determines the angle at the elbow joint. Each image is labeled and displayed with the calculated angle and visualized pose landmarks. If landmarks are not detected, it reports the absence for that image.

### Epochs and Batch Processing
- The training process is iterated over a specified number of epochs. In each epoch, all steps from latent space sampling to optimization are repeated until all pose detected correctly and feedback given

### Monitoring and Adjustments
- **Monitoring**: It's important to monitor the generator and discriminator losses to understand the training dynamics. Adjustments might be necessary if the model shows signs of instability (e.g., mode collapse, where the generator produces limited variety of outputs).
- **Adjustments**: Parameters such as the learning rate, batch size, and number of training epochs may be tuned based on the observed performance and training dynamics.

This detailed training process helps in guiding the development and optimization of the Pose estimation and correction model. 

# Results

Comparison Of Real & Generated Epileptic Data
![image](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/7db704e6-f3bb-4be1-b4f8-861d6717bfa8)

Comparison Of Real & Generated Non- Epileptic Data
![download (1) (2)](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/126694480/6a99207d-e75e-4949-a0c7-38e782d8f778)






![download (2)](https://github.com/saumitkunder/EEG-DATA-SYNTHESIS-USING-GANS/assets/109196162/02dc7215-9cd1-456d-b2b3-93fab61160b8)



# Future Work

3D Pose Estimation: Using 2D photos to extend the model
to estimate 3D postures is an interesting avenue for further
study. Although 2D posture estimate gives useful geographical
information, 3D pose estimation can give much more detailed
insights into how people move and interact with one another.
Applications for 3D pose estimation can be found in fields
including virtual reality, motion capture, and human motion
analysis. But handling uncertainties in posture estimate and

recovering depth information from 2D photos provide addi-
tional hurdles.

Robustness to Occlusions and Clutter: To make the model
function better in real-world circumstances, it must be made
more resilient to occlusions, cluttered backdrops, and changes
in illumination. The accuracy of posture estimation methods

can be considerably reduced by occlusions and clutter, par-
ticularly in crowded areas or scenes with intricate backdrops.

Future work can concentrate on creating methods to deal with
occlusions, separate people from the background, and adjust
to different lighting scenarios, all of which will increase the
resilience and dependability of the model.
Transfer Learning and Fine-Tuning: To adapt the model
to new tasks or domains with little annotated data, transfer
learning techniques can be investigated. Less labeled data can
nevertheless yield better results for researchers if they use
pre-trained models and refine them on target datasets. By
addressing the problem of data scarcity in posture estimation
tasks, transfer learning can improve the model’s ability to
generalize to new surroundings, poses, or people.

# Conclusion

Human Activity Recognition: Complex human actions or

gestures can be recognized from video sequences by com-
bining posture estimation and action recognition. The system

is capable of identifying and categorizing a range of human
actions, including running, walking, jumping, and interacting

with objects, by examining temporal sequences of position es-
timations. Applications for human activity recognition include

healthcare, human-computer interaction, and surveillance. It
makes it possible for intelligent systems to comprehend and
react to human behavior.
Pose Estimation in Unconstrained Environments: Another
crucial area for further study is adapting the model to operate
in unconstrained settings, including outside scenes. Additional

difficulties in unconstrained situations include different back-
grounds, illumination, and camera angles. To tackle these

obstacles, resilient algorithms are needed that can adjust to
various and fluctuating environmental circumstances, hence
broadening the model’s relevance to a more extensive array
of real-life situations.

### Achievements
- **Successful Pose Localized**: The MediaPipe model was successfully adapted and implemented for the specific task of detection landmarks and estimating the pose, showing good performance in correct detection of gym pose and providing feedback to the user to improve his/her posture.


