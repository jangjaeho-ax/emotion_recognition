import cv2
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import rotate3dPoint as rotate3d

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
def flatten(p1):
    lpow = math.pow(p1[0],2)+math.pow(p1[1],2)+math.pow(p1[2],2)
    alpha =int(math.pow(p1[2],2)/2)
    p2 = (math.pow(p1[0],2)-alpha,math.pow(p1[1],2)-alpha,0)
    return p2

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

            # See where the user's head tilting
            if y_angle < -10:
                text = "Looking Left"
            elif y_angle > 10:
                text = "Looking Right"
            elif x_angle < -10:
                text = "Looking Down"
            elif x_angle > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            #noss_tip =(int(nose_3d[0]),int(nose_3d[1]),int(nose_3d[2]))

            l = img_w *0.75
            if math.pow(l, 2) < math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2):
                l = math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2) + 10
            p1_z = int(nose_3d[2])
            p2_z = int(math.sqrt(math.pow(l, 2) - math.pow(p2[0] - p1[0], 2) - math.pow(p2[1] - p1[1], 2)) + p1_z)
            print('l',l)
            print('x',p2[0])
            print('y',p2[1])
            print('z',p2_z)


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

        '''
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
        '''
        #변환 전 3D landmark를 2차원 landmark로 표현
        show_scatter3D(coords)
        front_coords = coords

        for i in front_coords:
            i = flatten(i)
        #show_scatter(front_coords)


        '''
        rotated_coords=coords
        center_point = (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])), int(np.mean(coords[:, 2])))

        rotate3d.rotateX(rotated_coords, center_point, pitch)
        show_scatter(rotated_coords)
        rotate3d.rotateY(rotated_coords, center_point, yaw)
        show_scatter(rotated_coords)
        '''
        # plot coordinates. Use image's Y coordinate in z direction.
        # needs to be inverted as image coordinates start from top.
        # color by distance to camera.
        '''
        fig = plt.figure(figsize=[4, 4])

        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.scatter(rotated_coords[:, 0], rotated_coords[:, 2], -rotated_coords[:, 1], c=rotated_coords[:, 2],
                   cmap="PuBuGn_r", clip_on=False, vmax=2 * rotated_coords[:, 2].max())        

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

        #plt.show()
        '''
        cv2.waitKey()

    cv2.imshow('Head Pose Estimation', image)




    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()