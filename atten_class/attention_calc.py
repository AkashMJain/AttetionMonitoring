from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


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


class facial_point_operation:
    """docstring for facial_point_operation"""

    def init(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./atten_class/shape_predictor_68_face_landmarks.dat')
        self.MOUTH_AR_THRESH = 0.30
        self.MOUTH_AR_CONSEC_FRAMES = 20
        self.YAWN = 0
        self.COUNTER_1 = 0
        self.COUNTER_2 = 0
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 48
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (self.jStart, self.jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

    def loop_operation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.imshow("frame", gray)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            mouth = shape[self.mStart:self.mEnd]
            nose = shape[self.nStart:self.nEnd]
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            mouth = shape[self.mStart:self.mEnd]
            nose = shape[self.nStart:self.nEnd]

            mar = mouth_aspect_ration(mouth)
            lef_dist, rig_dist = face_movement(leftEye, rightEye, nose)

            nose = shape[self.nStart:self.nEnd]

            mouthEyeHull = cv2.convexHull(mouth)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [mouthEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < self.EYE_AR_THRESH:
                self.COUNTER_1 += 1
                if self.COUNTER_1 >= self.EYE_AR_CONSEC_FRAMES:
                    return 1
                    # cv2.putText(frame, "Eye Closing Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.COUNTER_1 = 0
                return 0

            if mar > self.MOUTH_AR_THRESH:
                self.COUNTER_2 += 1
                if self.COUNTER_2 >= self.MOUTH_AR_CONSEC_FRAMES:
                    return 2
                    # cv2.putText(frame, "Yawining Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                self.COUNTER_2 = 0
                return 0


# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--predictor", required=True, help="path to facial landmark predictor")
# ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
# args = vars(ap.parse_args())

# vs = VideoStream(src=args["webcam"]).start()
# time.sleep(1.0)


#     while True:
#     # frame = vs.read()
#     # frame = imutils.resize(frame, width=700)


#         if ear < EYE_AR_THRESH:
#             COUNTER_1 += 1
#             if COUNTER_1 >= EYE_AR_CONSEC_FRAMES:
#                 cv2.putText(frame, "Eye Closing Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             COUNTER_1 = 0

#         if mar > MOUTH_AR_THRESH:
#             COUNTER_2 += 1

#             if COUNTER_2 >= MOUTH_AR_CONSEC_FRAMES:
#                 cv2.putText(frame, "Yawining Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         else:
#             COUNTER_2 = 0

#         cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, "MAR: {:.2f}".format(mar), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, "Left_dist: {:.2f}".format(lef_dist), (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, "rig_dist: {:.2f}".format(rig_dist), (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # show the frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break

# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()
