from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import calculate_face_attention.attention_calc as ac


# class face_array:
#     def face_array():
#         print("print this if pass through this ")

#     def read_frame_fast(self):
#         video_file = FileVideoStram(args["video"]).start()
#         stime.sleep(1.0)

#         while video_file.more():
#             # grab the frame from the video_file
#             frame = video_file.read()
#             frame = imutils.resize(frame, width=450)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame = np.desta


class face_array(object):
    """docstring for face_array1"""

    def argument_init(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", required=True, help="path to video file")
        ap.add_argument("-p", "--prototxt", required=True, help="path to model")
        ap.add_argument("-w", "--weights", required=True, help="path to pre-trained weights")
        ap.add_argument("-t", "--threshold", type=float, default=0.5, help="max threshold value for detection")
        self.args = vars(ap.parse_args())

    # def calculate_face_attentiveness_for_each_face():

    def read_frame_fast(self):
        video_file = FileVideoStream(self.args["video"]).start()
        time.sleep(1.0)
        fps = FPS().start()

        while video_file.more():
            # grab the frame from the video_file
            frame = video_file.read()
            self.frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            gray = np.dstack([gray, gray, gray])

            # cv2.putText(gray, "Queue Size: {}".format(video_file.Q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if self.COUNTER == 5:
                self.face_array()
                # for i in range(0, len(self.faces)):

                # self.attention_calculation()
                self.COUNTER = 0
            else:
                self.COUNTER = self.COUNTER + 1

            # for i in range(0, len(self.faces)):
                # if self.eye[i] == 4:
                #     self.msg[i] = "eye_closed"
                # elif self.mouth[i] == 6:
                #     self.msg[i] = "mouth_open"

            cv2.imshow("Frame", self.frame)
            # cv2.imshow("faces", self.faces[0])
            # for face in self.faces:
            #     cv2.imshow("frame", face)

            for i in range(0, len(self.faces)):
                cv2.imshow("faces" + format(i + 1), self.faces[i])

            # cv2.waitKey(1)
            fps.update()

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        fps.stop()
        print("elapsed time : {:.2f}".format(fps.elapsed()))
        print("FPS  : {:.2f}".format(fps.fps()))

    def face_array(self):
        self.faces = []
        (h, w) = self.frame.shape[: 2]
        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detecions = self.net.forward()
        for face in range(0, detecions.shape[2]):
            conf = detecions[0, 0, face, 2]
            if conf < self.args["threshold"]:
                continue

            box = detecions[0, 0, face, 3: 7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 255, 0), 1)

            # frame = window[sY - 10:eY + 10, sX - 10:eX + 10]

            temp = self.frame[startY - 10:endY + 10, startX - 10:endX + 10]
            self.faces.append(temp)
            # print(self.faces)
        # return frame

    def attention_calculation(self):
        pass
        # frame = self.faces[]

        # for i in range(0, len(self.faces)):
        # cv2.imshow("face" + str(i), self.faces[i])
        # self.val[i] = self.obj.loop_operation(self.faces[i])
        # if self.val[i] == 0:
        #     self.eye[i] = self.atten[i] + 1
        # elif self.val[i] == 1:
        #     self.mouth[i] = self.eye[i] + 1
        # elif self.val[i] == 2:
        #     self.mouth[i] = self.mouth[i] + 1

        # print("NOT DETECTED")

    def __init__(self):
        super(face_array, self).__init__()
        self.COUNTER = 5
        self.eye = [100] * 0
        self.mouth = [100] * 0
        self.val = [4] * 0
        self.atten = [100] * 0
        self.obj = ac.facial_point_operation()
        self.obj.init()
        # self.arg = arg
        self.argument_init()
        self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["weights"])
        self.read_frame_fast()
        # print("well print this")


face_array()
