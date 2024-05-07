import onnxruntime
import cv2 as cv
import numpy as np

def draw_hand(img, kpts):
    colors = [
        (255, 255, 255),
        (0, 0, 204), (0, 0, 179), (0, 0, 230), (0, 0, 255),
        (0, 204, 163), (0, 179, 143), (0, 230, 184), (0, 255, 204),
        (82, 204, 0), (71, 179, 0), (92, 230, 0), (102, 255, 0),
        (204, 82, 0), (179, 71, 0), (230, 92, 0), (255, 102, 0),
        (204, 0, 163), (179, 0, 143), (230, 0, 184), (255, 0, 204)
    ]

    kpts = np.array(kpts).reshape((-1, 2))
    for pt, color in zip(kpts, colors):
        cv.circle(img, (int(pt[0]), int(pt[1])), 1, color, 2)

    parents = [ -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19 ]
    for i in range(1, len(kpts)):
        pt1 = (int(kpts[i][0]), int(kpts[i][1]))
        pt2 = (int(kpts[parents[i]][0]), int(kpts[parents[i]][1]))
        cv.line(img, pt1, pt2, colors[i], 2)

session = onnxruntime.InferenceSession(r'./onnx/hand_mobileone-s1_dsntrle-192x192.onnx')

cap = cv.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_h, frame_w = frame.shape[:2]
        roi = ((frame_w-256) // 2, (frame_h-256) // 2, 256, 256)
        cropped_frame = frame[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]].copy()

        img_resized = cv.resize(cropped_frame, (192, 192))
        img_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB).astype(np.float32)
        img_normalized = ((img_rgb - [127.5, 127.5, 127.5]) / [255.0, 255.0, 255.0])
        img_normalized = np.expand_dims(img_normalized.astype(np.float32), 0)
        net_in = np.ascontiguousarray(np.transpose(img_normalized, (0, 3, 1, 2)))
        input_name = session.get_inputs()[0].name

        output = session.run(['coordinates'], {input_name: net_in})[0][0]

        kpts = []

        for xys in output:
            x, y, _, _ = xys
            x = x * cropped_frame.shape[1]
            y = y * cropped_frame.shape[0]
            kpts.extend([x, y])

        draw_hand(cropped_frame, kpts)

        frame = frame // 2
        frame[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]] = cropped_frame

        cv.imshow('video', frame)
        key = cv.waitKey(1)
        if key == 27:
            break
    else:
        break

cap.release()
cv.destroyAllWindows()