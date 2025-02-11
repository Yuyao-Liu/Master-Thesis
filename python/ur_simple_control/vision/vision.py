import cv2
from multiprocessing import Queue
import numpy as np

def processCamera(device, args, cmd, queue : Queue):
    camera = cv2.VideoCapture(0)
    # if you have ip_webcam set-up you can do this (but you have put in the correct ip)
    #camera = cv2.VideoCapture("http://192.168.219.153:8080/video")

    try:
        while True:
            (grabbed, frame) = camera.read()
        #    frame = cv2.flip(frame, 1)
            scale_percent = 100 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("camera", frame)
            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord("q"):
                break

            queue.put_nowait({"x" : np.random.randint(0, 10),
                              "y" : np.random.randint(0, 10)})
    except KeyboardInterrupt:
        if args.debug_prints:
            print("PROCESS_CAMERA: caught KeyboardInterrupt, i'm out")
