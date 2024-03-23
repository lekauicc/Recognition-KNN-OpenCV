#-----获取人脸样本-----
import cv2
import socket
import time
#调用视频流
# cap = cv2.VideoCapture("http://192.168.137.235:8081//stream")
cap = cv2.VideoCapture(0)
#调用人脸分类器
face_detector = cv2.CascadeClassifier(r'haarcascades\haarcascade_frontalface_alt2.xml')  
# socket UDP 的创建
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,0)
try:
    while True:
        success,img=cap.read()
        s.sendto(img, ("192.168.137.1", 9090))  # 向服务器发送图像数据
        time.sleep(0.1)
except:
    pass
# finally:
#     pass
