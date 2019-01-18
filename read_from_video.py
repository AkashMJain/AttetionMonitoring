from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import os
import cv2
import atten_class.attention_calc as tp

obj = tp.facial_point_operation()
obj.init()


class video_data:

    def atten(self, frame):
        str = obj.loop_operation(frame)
        if str == 1:
            return "non_attentive_eyes_closed"
        elif str == 2:
            return "non_attentive_open_mouth"
        elif str == 0:
            return "attentive"
        else:
            return "not_dected"
        # self function is for argument declaration which we'll use later

    def argument_declaration(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", required=True,
                        help="path to input video file")
        ap.add_argument("-p", "--prototext", required=True, help="path to .deploy file")

        ap.add_argument("-m", "--model", required="True", help="path to .caffe file")
        ap.add_argument("-c", "--conf", default=0.5, type=float, help="threshold value")

        self.args = vars(ap.parse_args())

    # self function defines network for convolution
    def define_net(self):
        self.net = cv2.dnn.readNetFromCaffe(self.args["prototext"], self.args["model"])

    # self is main code part
    def attention_monitoring(self):
        (height, width) = self.frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (0, 0, 255))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < self.args["conf"]:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (sX, sY, eX, eY) = box.astype("int")
            y = sY - 10 if sY - 10 > 10 else sY + 10

            crop_frame = self.frame[sY - 10:eY + 10, sX - 10:eX + 10]
            text = "{:.2f}%".format(conf * 100) + self.atten(crop_frame)

            cv2.rectangle(self.frame, (sX, sY), (eX, eY), (0, 255, 0), 2)
            cv2.putText(self.frame, text, (sX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # self function give's frames from video to process
    def read_frame_fast(self):
        # construct the argument parse and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-v", "--video", required=True,
        #                 help="path to input video file")
        # args = vars(ap.parse_args())

        # start the file video stream thread and allow the buffer to
        # start to fill
        self.argument_declaration()

        print("[INFO] starting video file thread...")
        fvs = FileVideoStream(self.args["video"]).start()
        time.sleep(1.0)

        # start the FPS timer
        fps = FPS().start()
        self.define_net()
        # loop over frames from the video file stream
        while fvs.more():
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale (while still retaining 3
            # channels)
            frame = fvs.read()
            self.frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            gray = np.dstack([gray, gray, gray])

            # display the size of the queue on the frame
            cv2.putText(gray, "Queue Size: {}".format(fvs.Q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Drosinwss Code starting from here

            self.attention_monitoring()

            # Drowsiness code end

            # show the frame and update the FPS counter
            cv2.imshow("Frame", self.frame)
            cv2.waitKey(1)
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        fvs.stop()


obj1 = video_data()
# obj1.argument_declaration()
obj1.read_frame_fast()
