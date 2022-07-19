import cv2
import math
import numpy as np
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import rotate3dPoint as rotate3d
import index

def get_facemesh_coords(landmark_list, img):
    """Extract FaceMesh landmark coordinates into 468x3 NumPy array.
    """
    #FAceMesh landmark에 좌표에 대응되는 ndarray를 구하는 함수
    h, w = img.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]

    #원 이미지의 높이, 넓이를 곱해줘야 원래 좌표의 크기가 나온다.
    return np.multiply(xyz, [w, h, w]).astype(int)

def show_scatter(coords):
    fig = plt.figure(figsize=[4, 4])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(coords[:, 0], -coords[:, 1], cmap="PuBuGn_r")
    ax.elev = -5
    ax.dist = 6
    ax.axis("off")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    max_range = np.array([np.diff(xlim), np.diff(ylim)]).max() / 2.0


    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    plt.show()
def show_scatter3D(coords):
    fig = plt.figure(figsize=[4, 4])
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.scatter(coords[:, 0], coords[:, 2], -coords[:, 1], c=coords[:, 2],
               cmap="PuBuGn_r", clip_on=False, vmax=2 * coords[:, 2].max())
    ax.elev = -5
    ax.dist = 6
    ax.axis("off")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    max_range = np.array([np.diff(xlim), np.diff(ylim), np.diff(zlim)]).max() / 2.0

    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
def get_angle(v1, v2):
    cos = np.dot(v1,v2)/norm(v1)/norm(v2)
    angle = math.degrees(np.arccos(np.clip(cos, -1, 1)))
    return angle
def eval_left_eye(_landmark, img_shape):
    img_h, img_w, img_c= img_shape

    p33 = np.array([int(_landmark[33].x * img_w), int(_landmark[33].y * img_h), int(_landmark[33].z * img_w)])
    p159 = np.array([int(_landmark[159].x * img_w), int(_landmark[159].y * img_h), int(_landmark[159].z * img_w)])
    p133 = np.array([int(_landmark[133].x * img_w), int(_landmark[133].y * img_h), int(_landmark[133].z * img_w)])
    p145 = np.array([int(_landmark[145].x * img_w), int(_landmark[145].y * img_h), int(_landmark[145].z * img_w)])

    v159to33 = p33 - p159
    v159to133 = p133 - p159
    v145to33 = p33 - p145
    v145to133 = p133 - p145
    v145to159 = p159 -p145

    print('left eye-------------------------')
    print('ltop', get_angle(v145to159, v159to33))
    print('rtop', get_angle(v145to159, v159to133))
    print('lbottom', get_angle(v145to159, v145to33))
    print('rbottom', get_angle(v145to159, v145to133))
    return
def eval_right_eye(_landmark, img_shape):
    img_h, img_w, img_c = img_shape

    p362 = np.array([int(_landmark[362].x * img_w), int(_landmark[362].y * img_h), int(_landmark[362].z * img_w)])
    p386 = np.array([int(_landmark[386].x * img_w), int(_landmark[386].y * img_h), int(_landmark[386].z * img_w)])
    p263 = np.array([int(_landmark[263].x * img_w), int(_landmark[263].y * img_h), int(_landmark[263].z * img_w)])
    p374 = np.array([int(_landmark[374].x * img_w), int(_landmark[374].y * img_h), int(_landmark[374].z * img_w)])

    v386to363 = p362 - p386
    v386to263 = p263 - p386
    v374to362 = p362 - p374
    v374to263 = p263 - p374
    v374to386 = p386 - p374

    print('right eye------------------------')
    print('ltop', get_angle(v374to386, v386to363))
    print('rtop', get_angle(v374to386, v386to263))
    print('lbottom', get_angle(v374to386, v374to362))
    print('rbottom', get_angle(v374to386, v374to263))
    return
def eval_left_brow(_landmark,img_shape):
    img_h, img_w, img_c = img_shape

    p70 = np.array([int(_landmark[70].x * img_w), int(_landmark[70].y * img_h), int(_landmark[70].z * img_w)])
    p105 = np.array([int(_landmark[105].x * img_w), int(_landmark[105].y * img_h), int(_landmark[105].z * img_w)])
    p107 = np.array([int(_landmark[107].x * img_w), int(_landmark[107].y * img_h), int(_landmark[107].z * img_w)])
    p52 = np.array([int(_landmark[52].x * img_w), int(_landmark[52].y * img_h), int(_landmark[52].z * img_w)])

    v105to70 = p70 - p105
    v105to107 = p107 - p105
    v105to52 = p52 - p105

    print('left brow------------------------')
    print('ltop : ', get_angle(v105to70, v105to52))
    print('rtop : ', get_angle(v105to107, v105to52))
    return
def eval_right_brow(_landmark,img_shape):
    img_h, img_w, img_c = img_shape

    p336 = np.array([int(_landmark[336].x * img_w), int(_landmark[336].y * img_h), int(_landmark[336].z * img_w)])
    p334 = np.array([int(_landmark[334].x * img_w), int(_landmark[334].y * img_h), int(_landmark[334].z * img_w)])
    p300 = np.array([int(_landmark[300].x * img_w), int(_landmark[300].y * img_h), int(_landmark[300].z * img_w)])
    p282 = np.array([int(_landmark[282].x * img_w), int(_landmark[282].y * img_h), int(_landmark[282].z * img_w)])

    v334to335 = p336 - p334
    v334to300 = p300 - p334
    v334to282 = p282 - p334

    print('left brow------------------------')
    print('ltop : ', get_angle(v334to335, v334to282))
    print('rtop : ', get_angle(v334to300, v334to282))
    return

def eval_left_mouth(_landmark, img_shape):
    img_h, img_w, img_c = img_shape
    p61 = np.array([int(_landmark[61].x * img_w), int(_landmark[61].y * img_h), int(_landmark[61].z * img_w)])
    p13 = np.array([int(_landmark[13].x * img_w), int(_landmark[13].y * img_h), int(_landmark[13].z * img_w)])
    p291 = np.array([int(_landmark[291].x * img_w), int(_landmark[291].y * img_h), int(_landmark[291].z * img_w)])
    p14 = np.array([int(_landmark[14].x * img_w), int(_landmark[14].y * img_h), int(_landmark[14].z * img_w)])

    v13to61 = p61 - p13
    v14to61 = p61 - p14
    v61to291 = p291 - p61
    v13to14 = p14 -p13
    gap_ratio = norm(v13to14)/norm(v61to291)
    print('left mouth-----------------------')
    print('top : ', get_angle(v13to14, v13to61))
    print('bottom : ', get_angle(v13to14, v14to61))
    print('gap : ', gap_ratio)
    return
def eval_right_mouth(_landmark, img_shape):
    img_h, img_w, img_c = img_shape
    p61 = np.array([int(_landmark[61].x * img_w), int(_landmark[61].y * img_h), int(_landmark[61].z * img_w)])
    p13 = np.array([int(_landmark[13].x * img_w), int(_landmark[13].y * img_h), int(_landmark[13].z * img_w)])
    p291 = np.array([int(_landmark[291].x * img_w), int(_landmark[291].y * img_h), int(_landmark[291].z * img_w)])
    p14 = np.array([int(_landmark[14].x * img_w), int(_landmark[14].y * img_h), int(_landmark[14].z * img_w)])
    p87 = np.array([int(_landmark[87].x * img_w), int(_landmark[87].y * img_h), int(_landmark[87].z * img_w)])
    v13to291 = p291 - p13
    v14to291 = p87 - p14
    v61to291 = p291 - p61
    v13to14 = p14 - p13
    gap_ratio = norm(v13to14) / norm(v61to291)
    print('right mouth-----------------------')
    print('top : ', get_angle(v13to14, v13to291))
    print('bottom : ', get_angle(v13to14, v14to291))
    print('gap : ', gap_ratio)
    return

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1),cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)



    img_h, img_w, img_c = image.shape

    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        coords = get_facemesh_coords(results.multi_face_landmarks[0], image)

        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_w)



                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])
            landmark_dict = {}



                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            proj_mat = np.hstack((rmat, trans_vec))
            euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]

            pitch, yaw, roll = [math.radians(_) for _ in euler_angles]


            # Get the y rotation degree
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360
            for idx in index.INDEX_LIST:
                landmark_dict[idx] = results.multi_face_landmarks[0].landmark[idx]
            # See where the user's head tilting
            if y_angle < 0:
                text = "Looking Left"
                eval_right_eye(landmark_dict,image.shape)
                #eval_right_brow(landmark_dict,image.shape)
                eval_right_mouth(landmark_dict,image.shape)

            elif y_angle >= 0:
                text = "Looking Right"
                eval_left_eye(landmark_dict,image.shape)
                #eval_left_brow(landmark_dict,image.shape)
                eval_left_mouth(landmark_dict,image.shape)


            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            #noss_tip =(int(nose_3d[0]),int(nose_3d[1]),int(nose_3d[2]))



            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        # print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)



    cv2.imshow('Head Pose Estimation', image)
    cv2.waitKey()



    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()