import cv2
import mediapipe as mp
import numpy as np
import time
import pyramids
import fastfouriertransform
import hrcalculator


def tracking():

    vid = cv2.VideoCapture(1)
    ptime = 0
    # fps = int(vid.get(cv2.CAP_PROP_FPS))
    # print(fps)
    video_frames = []

    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mpdraw = mp.solutions.drawing_utils
    print('press q for results')
    while True:
        ret, frame = vid.read()
        # print(frame.shape)

        if ret:

            h, w, c = frame.shape
            imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_frame = frame
            result = hands.process(imgrgb)
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime

            if result.multi_hand_landmarks:

                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for mh in result.multi_hand_landmarks:

                    for id, lm in enumerate(mh.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx > x_max:
                            x_max = cx
                        if cx < x_min:
                            x_min = cx
                        if cy > y_max:
                            y_max = cy
                        if cy < y_min:
                            y_min = cy
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mpdraw.draw_landmarks(frame, mh, mphands.HAND_CONNECTIONS)
                    roi_frame = frame[y_min:y_min + y_max, x_min:x_min + x_max]
                    if roi_frame.size != frame.size:
                        # roi_frame.size=h1,w1,c1
                        roi_frame = cv2.resize(roi_frame, (500, 500))
                        frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                        frame[:] = roi_frame * (1. / 255)
                        video_frames.append(frame)
            frame_length = len(video_frames)

            #
            cv2.imshow('face', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
    vid.release()
    cv2.destroyAllWindows()

    return video_frames, frame_length, fps
