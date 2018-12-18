from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

# detector = None


# MAR methods


def face_movement(leftEye, rightEye, nose):
    A = dist.euclidean(leftEye[0], nose[0])
    B = dist.euclidean(rightEye[3], nose[0])
    return A, B


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ration(mouth):
    D1 = dist.euclidean(mouth[13], mouth[19])
    D2 = dist.euclidean(mouth[14], mouth[18])
    D3 = dist.euclidean(mouth[15], mouth[17])
    D4 = dist.euclidean(mouth[12], mouth[16])
    mar = (D1 + D2 + D3) / (3.0 * D4)
    return mar


# if __name__ == '__main__':
#     main()

class attention:
    # print("riched to attention")
    # global variable

    MOUTH_AR_THRESH = 0.30
    MOUTH_AR_CONSEC_FRAMES = 30
    # YAWN = 0
    COUNTER_1 = 0
    COUNTER_2 = 0
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 30

    def init(self):
        # __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        # print("Hello i am inside :)")
        self.detector = dlib.get_frontal_face_detector()

        self.predictor = dlib.shape_predictor(os.getcwd()+ '/atten_class/shape_predictor_68_face_landmarks.dat')

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (self.jStart, self.jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

    def calc_data(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # getting facial points for calculation

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]
            nose = shape[self.nStart:self.nEnd]

            # calling individual aspect ratio methods
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mar = mouth_aspect_ration(mouth)
            lef_dist, rig_dist = face_movement(leftEye, rightEye, nose)
            ear = (leftEAR + rightEAR) / 2.0
            # conditions

            if ear < self.EYE_AR_THRESH:
                self.COUNTER_1 += 1
                if self.COUNTER_1 >= self.EYE_AR_CONSEC_FRAMES:
                    return 2
                    # cv2.putText(frame, "Eye Closing Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.COUNTER_1 = 0
                return 1

            if mar > self.MOUTH_AR_THRESH:
                self.COUNTER_2 += 1
                if self.COUNTER_2 >= self.MOUTH_AR_CONSEC_FRAMES:
                    return 2

                    # cv2.putText(frame, "Yawining Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.COUNTER_2 = 0
                return 1

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # def test():
    #     print("test successfull")
