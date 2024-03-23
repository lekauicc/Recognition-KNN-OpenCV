import socket
import cv2
import io
from PIL import Image
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
s.bind(("", 8080))
while True:
    data, IP = s.recvfrom(400000)
    print("data的类型为："+str(type(data)))
    #解码
    nparr = np.fromstring(data, np.uint8)
    print("解码后的类型为："+str(type(nparr)))
    #解码成图片numpy
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("图片np的类型为："+str(type(img_decode)))
    print(img_decode)
    cv2.imwrite(r"face\getfaceby_comp\ "+"6"+'.'+"1"+'.jpg',img_decode) 
    cv2.imshow('result',img_decode)

    # bytes_stream = io.BytesIO(data)
    # image = Image.open(bytes_stream)
    # img = np.asarray(image)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # ESP32采集的是RGB格式，要转换为BGR（opencv的格式）
    # cv2.imshow("ESP32 Capture Image", img)
    # if cv2.waitKey(1) == ord("q"):
    #     break
