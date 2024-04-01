import numpy as np
import cv2 
from matplotlib import pyplot as plt
import os
from skimage import measure

# This is a simple program to try using basic image processing techniques to isolate the hand object.
# Process loosely based off of the paper: http://dx.doi.org/10.4236/jsip.2012.33047

plt.plot()

# Set directory 
directory = r"asl_alphabet_test\asl_alphabet_test"

# Loop through files in directory 
for filename in os.listdir(directory):
    # Load image
    f = os.path.join(directory, filename)
    og_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    cv2.imshow('Original image', og_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    # Binarize image and blur iamge 
    img = cv2.resize(og_img, (640, 480))
    ret,binary_image = cv2.threshold(img,70,255,0)
    img = cv2.resize(binary_image, (640, 480))

    img = cv2.medianBlur(img,5)
    img = cv2.GaussianBlur(img,(5,5),0)

    cv2.imshow('Binary image', img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    # Perform morphological transformations 
    kernel = np.ones((45, 45),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Morphed image', img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    # Get objects - this is not yet successful at correctly identifying objects 
    labels = measure.label(binary_image, connectivity=2, background=0)
    features = measure.regionprops(labels)
    print("I found %d objects in total." % (len(features)))

    # Create SimpleBlobDetector 
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs - this is also not yet successful at identifying objects 
    keypoints = detector.detect(img)
    print(f'Found {len(keypoints)} blobs')

    # Draw detected blobs 
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Blob Detection', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('output.png', img_with_keypoints)
