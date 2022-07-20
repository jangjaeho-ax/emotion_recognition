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
import mediapipe as mp
import my_function as mf

#emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] #Emotion list
emotions = ["anger", "neutral", "joy"] #Emotion list

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {} #Make dictionary for all values
face_3d_point = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float64)

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("train\%s\*" %emotion) #change dataset directory address here!
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

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
            image = cv2.imread(item)  # open image
            img_h, img_w, img_c = image.shape
            face_2d = []  # 랜드마크의 x,y 정보만 저장할 리스트

            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                for face_num, face_landmarks in enumerate(results.multi_face_landmarks):
                    for idx, lm in enumerate(face_landmarks.landmark):
                        x, y = float(lm.x * img_w), float(lm.y * img_h)
                        face_2d.append([x, y])
                    face_2d_point = np.array([
                        face_2d[1],
                        face_2d[152],
                        face_2d[226],
                        face_2d[446],
                        face_2d[57],
                        face_2d[287]
                    ], dtype="double")
                    # 중간 값을 계산하기 위한 최대 최소 값 구하기

                    # The camera matrix
                    focal_length = 1 * img_h
                    center = (img_h / 2, img_w / 2)
                    cam_matrix = np.array(
                        [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double"
                    )
                    maxXY = max(face_2d, key=mf.x_element)[0], max(face_2d, key=mf.y_element)[1]
                    minXY = min(face_2d, key=mf.x_element)[0], min(face_2d, key=mf.y_element)[1]

                    xcenter = (maxXY[0] + minXY[0]) / 2
                    ycenter = (maxXY[1] + minXY[1]) / 2
                    # 카메라 왜곡이 없다 가정
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # PnP 문제를 해결
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d_point, face_2d_point, cam_matrix, dist_matrix)

                    # 회전 행렬 구하기
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    proj_mat = np.hstack((rmat, trans_vec))
                    euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]

                    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
                    break
                coords = mf.get_facemesh_coords(results.multi_face_landmarks[0], image)
                rotated_coords = coords
                center_point = (int(xcenter), int(ycenter), int(((xcenter - img_w / 2) ** 2 + (ycenter - img_h / 2) ** 2) ** .4))
                mf.rotateX(rotated_coords, center_point, -(math.pi - pitch))
                dict = {}
                for r in rotated_coords:
                    dict[r[3]] = [r[0], r[1], r[2]]
                mf.flip_line_axis(dict, yaw)
                landmark2D = mf.change_to_2D(dict)
                #mf.show_scatter2(landmark2D)
                data = mf.vectorize_landmark(landmark2D)
                training_data.append(data['landmarks_vectorized'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))

            else:
                #print("no face detected on this one")
                fail_counter = fail_counter + 1
        #do the same for test dataset as above
        for item in prediction:
            image = cv2.imread(item)  # open image
            img_h, img_w, img_c = image.shape
            face_2d = []  # 랜드마크의 x,y 정보만 저장할 리스트

            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                for face_num, face_landmarks in enumerate(results.multi_face_landmarks):
                    for idx, lm in enumerate(face_landmarks.landmark):
                        x, y = float(lm.x * img_w), float(lm.y * img_h)
                        face_2d.append([x, y])
                    face_2d_point = np.array([
                        face_2d[1],
                        face_2d[152],
                        face_2d[226],
                        face_2d[446],
                        face_2d[57],
                        face_2d[287]
                    ], dtype="double")
                    # 중간 값을 계산하기 위한 최대 최소 값 구하기

                    # The camera matrix
                    focal_length = 1 * img_h
                    center = (img_h / 2, img_w / 2)
                    cam_matrix = np.array(
                        [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double"
                    )
                    maxXY = max(face_2d, key=mf.x_element)[0], max(face_2d, key=mf.y_element)[1]
                    minXY = min(face_2d, key=mf.x_element)[0], min(face_2d, key=mf.y_element)[1]

                    xcenter = (maxXY[0] + minXY[0]) / 2
                    ycenter = (maxXY[1] + minXY[1]) / 2

                    # 카메라 왜곡이 없다 가정
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # PnP 문제를 해결
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d_point, face_2d_point, cam_matrix, dist_matrix)

                    # 회전 행렬 구하기
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    proj_mat = np.hstack((rmat, trans_vec))
                    euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]

                    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
                    break
                coords = mf.get_facemesh_coords(results.multi_face_landmarks[0], image)

                rotated_coords = coords
                center_point = (int(xcenter), int(ycenter), int(((xcenter - img_w / 2) ** 2 + (ycenter - img_h / 2) ** 2) ** .4))
                mf.rotateX(rotated_coords, center_point, -(math.pi - pitch))
                dict = {}
                for r in rotated_coords:
                    dict[r[3]] = [r[0], r[1], r[2]]
                #ori_dict = dict
                landmark2D=mf.get_side_face(dict, yaw)
                mf.show_scatter2(landmark2D)

                #landmark2D = mf.change_to_2D(side_face)
                #ori_landmark2D = mf.change_to_2D(ori_dict)
                data = mf.vectorize_landmark(landmark2D)
                prediction_data.append(data['landmarks_vectorized'])  # append image array to training data list
                prediction_labels.append(emotions.index(emotion))

            else:
                #print("no face detected on this one")
                fail_counter = fail_counter + 1
        counter = counter + len(training) + len(prediction)

    print('counter : ', counter )
    print('fail_counter : ', fail_counter)
    return training_data, training_labels, prediction_data, prediction_labels   

accur_lin = []
for i in range(0,3): #set nunmber of traning iterations here
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
pkl_filename = "pickle_model2.pkl"
with open(pkl_filename, 'wb') as file:  
    pickle.dump(clf, file) #writes model to a pickle file. 

print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs

