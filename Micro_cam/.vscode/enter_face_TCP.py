from socket import *
import cv2
import io
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():

    port=8080

    # 创建TCPsocket
    tcp_server_socket = socket(AF_INET, SOCK_STREAM)
    # 本地信息
    address = ('0.0.0.0', port)
    # 绑定本地信息
    tcp_server_socket.bind(address)
    # 将主动套接字变为被动套接字
    tcp_server_socket.listen(128)
    print('等待用户连接。。。')
    client_socket, client_Addr = tcp_server_socket.accept()
    print(client_Addr)

    while True:
        # 等待客户端的链接，即为这个客户端发送文件
        # 接收对方发送过来的数据
        data = client_socket.recv(10000)  # 一次接收100k数据
        # print(data)
        bytes_stream = io.BytesIO(data)
        # print(bytes_stream)
        image = Image.open(bytes_stream)
        img = np.asarray(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # ESP32采集的是RGB格式，要转换为BGR（opencv的格式）
        cv2.imshow("ESP32 Capture Image", img)
        if cv2.waitKey(1) == ord("q"):
            break
    client_socket.close()
        

    # 关闭监听套接字
    tcp_server_socket.close()


if __name__ == "__main__":
    main()

