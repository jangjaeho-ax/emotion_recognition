import math
import numpy as np
import matplotlib.pyplot as plt
import index

def x_element(elem):
    return elem[0]
def y_element(elem):
    return elem[1]
def rotateX(np_array, center, radians):
    for p in np_array:
        y = p[1] - center[1]
        z = p[2] - center[2]
        d = math.hypot(y, z)
        theta = math.atan2(y, z) + radians
        p[2] = center[2] + d * math.cos(theta)
        p[1] = center[1] + d * math.sin(theta)
def rotateY(np_array, center, radians):
    for p in np_array:
        x = p[0] - center[0]
        z = p[2] - center[2]
        d = math.hypot(x, z)
        theta = math.atan2(x, z) + radians
        p[2] = center[2] + d * math.cos(theta)
        p[0] = center[0] + d * math.sin(theta)

def rotateZ(np_array, center, radians):
    #radians = m.radians(degree)
    for p in np_array:
        x = p[0] - center[0]
        y = p[1] - center[1]
        d = math.hypot(y, x)
        theta = math.atan2(y, x) + radians
        p[0] = center[0] + d * math.cos(theta)
        p[1] = center[1] + d * math.sin(theta)

def get_facemesh_coords(landmark_list, img):

    #FAceMesh landmark에 좌표에 대응되는 ndarray를 구하는 함수
    h, w = img.shape[:2]  # grab width and height from image
    xyzi = [(lm.x, lm.y, lm.z, idx) for idx, lm in enumerate(landmark_list.landmark)]

    #원 이미지의 높이, 넓이를 곱해줘야 원래 좌표의 크기가 나온다.
    return np.multiply(xyzi, [w, h, w, 1]).astype(float)

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
def show_scatter2(coords):
    fig = plt.figure(figsize=[4, 4])
    ax = fig.add_axes([0, 0, 1, 1])
    x_list = []
    y_list = []
    for c in coords:
        x_list.append(c[0])
        y_list.append(-c[1])
    ax.scatter(x_list, y_list, cmap="PuBuGn_r")
    ax.elev = 0
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

def change_to_2D(input_landmark):#3차원 랜드마크를 2차원 랜드마크 이미지로 변환
    result = []
    for idx in index.TO2D_FACE:
        result.append(input_landmark[idx])
    return result
def flip_line_axis(landmark, yaw): #랜드마크 이미지를 턱끝과 이마 중간을 기준 축으로 대칭 이동
    axis_point1 ,axis_point2 = landmark[151],landmark[152]
    dx, dy, dz = np.array(axis_point1) - np.array(axis_point2)
    w = dy / dx
    b = axis_point1[1] - w * axis_point1[0]

    if yaw >= 0.3:
        if dx != 0:
            for li, ri in zip(index.LEFT_FACE,index.RIGHT_FACE):
                new_x,new_y = get_symmetry_point(w,b,landmark[ri])
                landmark[li] = [new_x, new_y, landmark[li][2]]
                #landmark[li] = [int(new_x), int(new_y), landmark[li][2]]
        else:
            for li, ri in zip(index.LEFT_FACE,index.RIGHT_FACE):
                new_x = 2*axis_point1[0]-landmark[ri][0]
                new_y = landmark[ri][1]
                landmark[li] = [new_x, new_y, landmark[li][2]]
    elif yaw <= -0.3:
        if dx != 0:
            for li, ri in zip(index.LEFT_FACE,index.RIGHT_FACE):
                new_x,new_y = get_symmetry_point(w,b,landmark[li])
                landmark[ri] = [new_x, new_y, landmark[ri][2]]
        else:
            for li, ri in zip(index.LEFT_FACE, index.RIGHT_FACE):
                new_x = 2 * axis_point1[0] - landmark[li][0]
                new_y = landmark[li][1]
                landmark[li] = [new_x, new_y, landmark[ri][2]]
    return
def get_symmetry_point(w, b, point): #대칭 이동 함수
    x, y = point[0],point[1]
    new_x = x - 2 * w * (w * x - 1 * y + b) / (w**2 + 1)
    new_y = y - 2 * -1 * (w * x - 1 * y + b) / (w**2 + 1)
    return new_x, new_y
def vectorize_landmark(landmark): #랜드마크를 벡터화 시킴
    data = {}
    xlist = []
    ylist = []
    for i in range(0, 76):  # Store X and Y coordinates in two lists
        xlist.append(landmark[i][0])
        ylist.append(landmark[i][1])
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
    return data