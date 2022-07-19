import cv2
import math
import numpy as np
import time
import mediapipe as mp
import pickle
import my_function as mf
import glob

def test_emotion_recognition():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    emotions = ["anger", "neutral", "joy"]  # Emotion list
    file_paths = glob.glob("test\*")  # change dataset directory address here!
    # emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] #Emotion list
    pkl_filename = 'pickle_model2.pkl'  # trained model file
    with open(pkl_filename, 'rb') as file:  # load all weights from model
        pickle_model = pickle.load(file)

    face_3d_point = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=np.float64)
    start = time.time()
    for file_path in file_paths:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        print(file_path[5:])

        # To improve performance
        image.flags.writeable = False
        # Get the result
        results = face_mesh.process(image)
        img_h, img_w, img_c = image.shape
        face_2d = []  # 랜드마크의 x,y 정보만 저장할 리스트
        face_2d_point = []  # 포즈 인식을 위한 투영점 저장 리스트
        if results.multi_face_landmarks:
            dist = []
            coords = mf.get_facemesh_coords(results.multi_face_landmarks[0], image)  # 다루기 쉽도록 넘파이 어레이로 변환
            for face_num, face_landmarks in enumerate(results.multi_face_landmarks):
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = float(lm.x * img_w), float(lm.y * img_h)
                    face_2d.append([x, y])
                # Convert it to the NumPy array
                face_2d_point = np.array([
                    face_2d[1],
                    face_2d[152],
                    face_2d[226],
                    face_2d[446],
                    face_2d[57],
                    face_2d[287]
                ], dtype="double")
                # 중간 값을 계산하기 위한 최대 최소 값 구하기
                maxXY = max(face_2d, key=mf.x_element)[0], max(face_2d, key=mf.y_element)[1]
                minXY = min(face_2d, key=mf.x_element)[0], min(face_2d, key=mf.y_element)[1]

                xcenter = (maxXY[0] + minXY[0]) / 2
                ycenter = (maxXY[1] + minXY[1]) / 2

                # The camera matrix
                focal_length = 1 * img_h
                center = (img_h / 2, img_w / 2)
                cam_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double"
                )

                # 카메라 왜곡이 없다 가정
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # PnP 문제를 해결
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d_point, face_2d_point, cam_matrix, dist_matrix)

                # 회전 행렬 구하기
                rmat, jac = cv2.Rodrigues(rot_vec)

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec,
                                                                 trans_vec, cam_matrix, dist_matrix)

                p1 = (int(face_2d_point[0][0]), int(face_2d_point[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 2)

                proj_mat = np.hstack((rmat, trans_vec))
                euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]

                pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
            '''
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
            '''
            #cv2.imshow('Head Pose Estimation', image)
            #mf.show_scatter(coords)
            rotated_coords = coords
            center_point = (int(xcenter), int(ycenter), int(((xcenter - img_w / 2) ** 2 + (ycenter - img_h / 2) ** 2) ** .4))
            mf.rotateX(rotated_coords, center_point, -(math.pi - pitch))
            #mf.show_scatter(rotated_coords)

            rc_dict = {}
            for r in rotated_coords:
                rc_dict[r[3]] = [r[0], r[1], r[2]]
            # print(rc_dict)
            mf.flip_line_axis(rc_dict, yaw)
            landmark2D = mf.change_to_2D(rc_dict)
            #mf.show_scatter2(landmark2D)
            data = mf.vectorize_landmark(landmark2D)
            if data['landmarks_vectorized'] != "error":  # if landmarks are detected..
                prediction_data = np.array(data['landmarks_vectorized'])  # convert to numpy array ..
                predicted_labels = pickle_model.predict(prediction_data.reshape(1, -1))  # to get predicted values ...
                print('emotion : ', emotions[predicted_labels[0]])  # prints the predicted emotion
            else:
                print("error!!!!")
        else :
            print('Landmark not found')
    end = time.time()
    totalTime = end - start
    print('total time : ',totalTime)








