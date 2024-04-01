# before running: 
# conda install python=3.8
# conda install -c conda-forge numpy=1.21.6
# pip install -q mediapipe
# wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# I also had to update/reinstall several other packages (matplotlib, pillow, etc) based on errors 

# source: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=gxbHBsF-8Y_l

# This is an example for getting the landmark information for some images 

# Import the necessary modules.
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Method to draw landmarks on image
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    #cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


# Create an HandLandmarker object
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2, 
                                       min_hand_detection_confidence=0.01)
detector = vision.HandLandmarker.create_from_options(options)

# Directory to data 
#directory = r"asl_alphabet_test\asl_alphabet_test"
directory = r"asl_alphabet_train\asl_alphabet_train"

# Loops through subset of images in training directory (for efficiency)
# Shows the landmarked image and the number of images where a hand was found to get idea of success rate 
for folder in os.listdir(directory):
    count_found = 0
    for filename in os.listdir(os.path.join(directory, folder))[:30]:
      # Load the input image.
      f = os.path.join(directory, folder, filename)
      image = mp.Image.create_from_file(f)

      # Potential future preprocessing of images
      # Attempts to increase brightness and clarity so far decreased detection.  

      # Detect hand landmarks from the input image.
      detection_result = detector.detect(image)
      if len(detection_result.hand_landmarks) != 0:
        count_found += 1

      # Process the classification result and visualize it
      annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
      cv2.imshow(f'Landmarked Image: {f}', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
      cv2.waitKey(0)
      cv2.destroyAllWindows() 
    print(f'Found {count_found} hands out of 30 in {folder}')
