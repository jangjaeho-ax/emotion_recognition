import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
import time
from sklearn.svm import SVC
import pickle

#emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] #Emotion list
emotions = ["anger", "neutral", "joy"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {} #Make dictionary for all values


def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("train\%s\*" %emotion) #change dataset directory address here!
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        #record mean values of both X Y coordinates    
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        #store central deviance 
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorized = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):#analysing presence of facial landmarks 
            landmarks_vectorized.append(w)
            landmarks_vectorized.append(z)
            #extract center of gravity with mean of axis
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            #measuring distance and angle of each landmark from center of gravity 
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorized.append(dist)
            landmarks_vectorized.append((math.atan2(y, x)*360)/(2*math.pi))
        
        data['landmarks_vectorized'] = landmarks_vectorized#store landmarks in global dictionary
    if len(detections) < 1: #if no landmarks were detected, store error in dictionary 
        data['landmarks_vectorized'] = "error"

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    counter =0
    fail_counter = 0
    for emotion in emotions: #train for each emotion 
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion) #obtain the dataset
        
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item,cv2.IMREAD_GRAYSCALE) #open image
            get_landmarks(image) #extract landmarks
            if data['landmarks_vectorized'] == "error":
                print("no face detected on this one")
                fail_counter = fail_counter + 1
            else:
                training_data.append(data['landmarks_vectorized']) #append image array to training data list
                training_labels.append(emotions.index(emotion))

        #do the same for test dataset as above
        for item in prediction:
            image = cv2.imread(item, cv2.IMREAD_GRAYSCALE)

            get_landmarks(image)
            if data['landmarks_vectorized'] == "error":
                print("no face detected on this one")
                fail_counter = fail_counter + 1
            else:
                prediction_data.append(data['landmarks_vectorized'])
                prediction_labels.append(emotions.index(emotion))
        counter = counter + len(training) + len(prediction)

    print('counter : ', counter )
    print('fail_counter : ', fail_counter)
    return training_data, training_labels, prediction_data, prediction_labels   

accur_lin = []
for i in range(0,4): #set nunmber of traning iterations here
    start = time.time()
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    print(training_labels)
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)

    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# Create an variable to pickle and open it in write mode
pkl_filename = "pickle_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(clf, file) #writes model to a pickle file. 

print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs

