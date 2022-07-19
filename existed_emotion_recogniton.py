#Import required modules
import cv2
import dlib
import pickle
import numpy as np
import math
import glob

data = {} #Make dictionary for all values
data['landmarks_vectorized'] = [] #assign a key value to record landmarks
emotions = ["anger", "neutral", "joy"] #Emotion list
detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark identifier.
def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        # record mean values of both X Y coordinates
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        # store central deviance
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorized = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):  # analysing presence of facial landmarks
            landmarks_vectorized.append(w)
            landmarks_vectorized.append(z)
            # extract center of gravity with mean of axis
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            # measuring distance and angle of each landmark from center of gravity
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorized.append(dist)
            landmarks_vectorized.append((math.atan2(y, x) * 360) / (2 * math.pi))

        data['landmarks_vectorized'] = landmarks_vectorized  # store landmarks in global dictionary
    if len(detections) < 1:  # if no landmarks were detected, store error in dictionary
        data['landmarks_vectorized'] = "error"

def test_emotion_recognition():
    # Set up some required objects
    pkl_filename = 'pickle_model.pkl'  # trained model file
    with open(pkl_filename, 'rb') as file:  # load all weights from model
        pickle_model = pickle.load(file)
    file_paths = glob.glob("test\*")  # change dataset directory address here!
    for file_path in file_paths:
        image = cv2.imread(file_path)
        print(file_path[5:])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale as our dataset was grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))  # Does Local adapative histogram equalization for improved feed
        clahe_image = clahe.apply(gray)  # applies LAHE
        get_landmarks(clahe_image)  # obtain landmarks from input feed
        if data['landmarks_vectorized'] != "error":  # if landmarks are detected..
            prediction_data = np.array(data['landmarks_vectorized'])  # convert to numpy array ..
            predicted_labels = pickle_model.predict(prediction_data.reshape(1, -1))  # to get predicted values ...
            print('emotion : ', emotions[predicted_labels[0]])  # prints the predicted emotion
        else:
            print("landmark not found")



