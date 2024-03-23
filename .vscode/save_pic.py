import cv2

# 1.创建VideoCapture对象-->用于读取视频
# 1）读取摄像头
# cap = cv2.VideoCapture(0)
# 2）视频流地址：http://ivi.bupt.edu.cn/hls/cctv1hd.m3u8
cap = cv2.VideoCapture("http://192.168.43.140:81/stream")
# 3）视频文件路径：D:/迅雷下载/诛仙.mkv
# cap = cv2.VideoCapture("D:/迅雷下载/诛仙.mkv")

if not cap.isOpened():  # 判断cap是否成功捕获
    print("Cannot open camera")
    exit()

# 2.定义解码器并创造VideoWrite对象-->用于保存视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 定义解码器
# 创建视频流写入对象VideoWriter，VideoWriter_fourcc为视频编解码器，30为帧播放速率
# (1920, 1080)为视频帧大小，要与下面frame的大小一致(否则会导致保存的视频无法播放)
out = cv2.VideoWriter('D:/迅雷下载/output.avi', fourcc, 30, (1920, 1080))

while True:
    # 3.读取视频
    ret, frame = cap.read()  # ret表示是否成功获取帧，视频读到结尾，ret就为False。frame是每一帧的图像。

    if not ret:  # 如果未成功读取帧，就退出循环
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 4.显示视频图片
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)  # 创建一个窗口，可调整大小
    cv2.imshow("frame", frame)  # 展示视频图片

    # 5.保存视频图片
    # frame = cv2.flip(frame, 0)  # 沿x轴方向翻转图片并保存
    out.write(frame)  #保存视频图片
    print(frame.shape)  # (1080, 1920, 3)

    # 按'q'键退出循环
    if cv2.waitKey(41) & 0xFF == ord('q'):
        break

cap.release()  # 释放VideoCapture对象
out.release()  # 释放VideoWriter对象
cv2.destroyAllWindows()  # 销毁所有窗口