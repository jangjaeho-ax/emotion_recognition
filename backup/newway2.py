import cv2
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import rotate3dPoint as rotate3d
from skimage import data, transform

def x_element(elem):
    return elem[0]


def y_element(elem):
    return elem[1]


def get_facemesh_coords(landmark_list, img):
    """Extract FaceMesh landmark coordinates into 468x3 NumPy array.
    """
    # FAceMesh landmark에 좌표에 대응되는 ndarray를 구하는 함수
    h, w = img.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]

    # 원 이미지의 높이, 넓이를 곱해줘야 원래 좌표의 크기가 나온다.
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


def flatten(p1):
    lpow = math.pow(p1[0], 2) + math.pow(p1[1], 2) + math.pow(p1[2], 2)
    alpha = int(math.pow(p1[2], 2) / 2)
    p2 = (math.pow(p1[0], 2) - alpha, math.pow(p1[1], 2) - alpha, 0)
    return p2


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

face_3d_point =np.array([],dtype = np.float64)
face_2d_point = np.array([
    (0.0, 0.0 ),  # Nose tip
    (0.0, -330.0),  # Chin
    (-225.0, 170.0),  # Left eye left corner
    (225.0, 170.0),  # Right eye right corne
    (-150.0, -150.0),  # Left Mouth corner
    (150.0, -150.0,)  # Right mouth corner
], dtype=np.float64)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    img_h, img_w, img_c = image.shape

    face_2d = []
    face_3d =[]

    if results.multi_face_landmarks:
        dist = []
        coords = get_facemesh_coords(results.multi_face_landmarks[0], image)
        for face_num, face_landmarks in enumerate(results.multi_face_landmarks):
            for idx, lm in enumerate(face_landmarks.landmark):
                x, y, z = int(lm.x * img_w), int(lm.y * img_h), int(lm.z*img_w)

                face_3d.append([x, y,z])

            # Convert it to the NumPy array
            face_3d_point = np.array([
                face_3d[1],
                face_3d[152],
                face_3d[226],
                face_3d[446],
                face_3d[57],
                face_3d[287]
            ], dtype="double")

            focal_length = 1 * img_h
            center = (img_h / 2, img_w / 2)
            cam_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            # 카메라 왜곡이 없다 가정
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d_point, face_2d_point, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec,
                                                             trans_vec, cam_matrix, dist_matrix)

            p1 = (int(face_2d_point[0][0]), int(face_2d_point[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            proj_mat = np.hstack((rmat, trans_vec))
            euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]

            rotated_coords = cv2.warpPerspective(coords, proj_mat, (img_w,img_h))


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
        show_scatter(rotated_coords)
        cv2.imshow('Head Pose Estimation', image)
        cv2.waitKey()


    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()