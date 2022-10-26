'''
Author: linin00
Date: 2022-10-26 23:52:01
LastEditTime: 2022-10-27 00:10:34
LastEditors: linin00
Description: 通过摄像头进行人体姿态识别，速度非常慢
FilePath: /pytorch-openpose/app/task2/humanRecognitionInRealTime.py

'''
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
sys.path.append("../..")
from src import model
from src import util
from src.body import Body
class Demo() :
  def __init__(self):
    self.body_estimation = Body('../../model/body_pose_model.pth')
  def recognition(self, oriImg):
    candidate, subset = self.body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    return canvas

if __name__ == "__main__":
  cap = cv2.VideoCapture(0)
  demo = Demo()
  while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
    img = cv2.flip(frame, 1)
    img = demo.recognition(img)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()