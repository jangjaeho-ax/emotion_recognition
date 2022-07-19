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