# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import cv2 as ob

face = ob.CascadeClassifier('C:\\Haar\\haarcascade_frontalface_default.xml')
eyes = ob.CascadeClassifier('C:\\Haar\\haarcascade_eye.xml')

def FaceDetect(grey,frame):
    f=face.detectMultiScale(grey,1.3,5)
    for (a,b,c,d) in f:
        ob.rectangle(frame,(a,b,),(a+c,b+d),(255,0,0),2)
        gr = grey[b:b+d,a:a+c]
        cl = frame[b:b+d,a:a+c]
        eye = eyes.detectMultiScale(gr,1.1,25)
        for(ea,eb,ec,ed) in eye:
            ob.rectangle(cl,(ea,eb),(ea+eb,ec+ed),(0,255,0),2)
    return frame

video = ob.VideoCapture(ob.CAP_DSHOW)

while True:
    __UnusedVar,frame = video.read()
    grey = ob.cvtColor(frame,ob.COLOR_BGR2GRAY)
    canvas = FaceDetect(grey, frame)
    ob.imshow('Video',canvas)
    if ob.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
ob.destroyAllWindows()

