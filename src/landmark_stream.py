import cv2
import time
import mediapipe as mp
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) 

# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt
def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
  try: 
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]

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

    return annotated_image
  
  except:
    return rgb_image
   
# based off https://medium.com/@oetalmage16/a-tutorial-on-finger-counting-in-real-time-video-in-python-with-opencv-and-mediapipe-114a988df46a
# and https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream 
class Landmarker():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), 
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, 
         min_hand_detection_confidence = 0.01, # same options used to landmark training data
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   # detect landmarks
   def detect_async(self, frame):
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   # close landmarker
   def close(self):
      self.landmarker.close()

def main():
    cap = cv2.VideoCapture(0)

    hand_landmarker = Landmarker()

    while True:
        # pull frame
        ret, frame = cap.read()
        # mirror frame
        frame = cv2.flip(frame, 1)

        hand_landmarker.detect_async(frame)
        #print(hand_landmarker.result)
        frame = draw_landmarks_on_image(frame,hand_landmarker.result)

        # display frame
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # release everything
    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

if __name__ == "__main__":
   main()  

