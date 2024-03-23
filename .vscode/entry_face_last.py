from socket import *
import cv2
import io
from PIL import Image
import numpy as np
import os

# id_dict={}
# Tatal_face_num=8

# def init():  # 将config文件内的信息读入到字典中
#     f = open(r'my_config.txt')
#     global Total_face_num
#     Total_face_num = int(f.readline())
#     for i in range(int(Total_face_num)):
#         line = f.readline()
#         id_name = line.split(' ')
#         id_dict[int(id_name[0])] = id_name[1]
#     f.close()


def get_face():
    face_detector = cv2.CascadeClassifier(r'C:\Users\MI\Desktop\face_dis\haarcascade_frontalface_default.xml')  
    #创建UDP套接字。若使用TCP，掉了一帧就要重新发送，延迟很大，而且会导致cam端结束拍摄
    s = socket(AF_INET, SOCK_DGRAM, 0)
    s.bind(("0.0.0.0", 8080))
    #照片计数变量
    count=0
    #为即将录入的脸标记一个id
    f=open('my_config.txt','r')
    # 'my_config.txt'文件的第一行为录入的人数
    face_id=int(f.readline())
    while True:
        #获取图片并将其转换为三维矩阵变量‘img’
        data, IP = s.recvfrom(100000)
        bytes_stream = io.BytesIO(data)
        image = Image.open(bytes_stream)
        img = np.asarray(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # ESP32采集的是RGB格式，要转换为BGR（opencv的格式）
        # cv2.imshow("ESP32 Capture Image", img)
        #转为灰度图片，减少程序符合，提高识别度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        '''detectMultiScale采用滑动窗口的方式选取人脸
            其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
            scaleFactor越大，检测的速度越快，但正确率可能有所下降；
            minNeighbors=5，代表迭代后检测到的一个人的人脸数目要大于5才会保存他的人脸数据
        '''
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
        #框选人脸，for循环保证一个能检测的实时动态视频流
        for (x, y, w, h) in faces:

            '''xy为左上角的坐标,w为宽，h为高，(x+w, y+w)是右下角的坐标(y轴正半轴是向下的)。用rectangle为人脸标记画框(y)
            
            '''
            cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))#采集一次人脸数据
            #成功框选则样本数增加
            count += 1  
            #保存图像，把灰度图片看成二维数组来检测人脸区域
            #(这里是建立了data的文件夹，当然也可以设置为其他路径或者调用数据库)
            cv2.imwrite(r"C:\Users\MI\Desktop\face_dis\face\my_face\ "+str(face_id)+'.'+str(count)+'.jpg',gray[y:y+h,x:x+w]) 
            #将人脸框出来
            cv2.imshow('image',img) 
        #录入人脸时按下“q”，强制退出录入·  
        if cv2.waitKey(1) == ord("q"):
            break
        elif count >= 10:
                break


def finish_promote():
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)

    # 2. 链接服务器
    tcp_client_socket.connect(("192.168.43.118", 7788))

    # 提示用户输入数据
    send_data ='0'

    # 3. 向服务器发送数据
    tcp_client_socket.send(send_data.encode("utf-8"))

    # 4. 关闭套接字
    tcp_client_socket.close()

#获取图片的ID
def get_images_and_labels(path,detector):
        image_paths = [os.path.join(path,f) for f in os.listdir(path)]
        #新建个list用于存放
        face_samples = []
        ids = []
    
        #遍历图片路径，导入图片和id添加到list中
        for image_path in image_paths:
    
            #通过图片路径将其转换为灰度图片
            img = Image.open(image_path).convert('L')
    
            #将图片转化为数组
            img_np = np.array(img,'uint8')
            #os.path.split(image_path)只以iamge_path的最后一个’\‘作为分割符，将iamge_path分为两个字符串
            if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
                continue
    
            #为了获取id，将图片和路径分裂并获取
            id = int(os.path.split(image_path)[-1].split(".")[0])
            faces = detector.detectMultiScale(img_np)
    
            #将获取的图片和id添加到list中
            for(x,y,w,h) in faces:
                face_samples.append(img_np[y:y+h,x:x+w])
                ids.append(id)
        return face_samples,ids

def train_face():
    #设置之前收集好的数据文件路径
    path = r"face\my_face"
    # print(cv2.__version__)#输出cv2的版本号
    #初始化识别的方法
    recog = cv2.face.LBPHFaceRecognizer_create()
    
    #调用熟悉的人脸分类器
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    #创建一个函数，用于从数据集文件夹中获取训练图片,并获取id
    #注意图片的命名格式为User.id.sampleNum
    
    
    #调用函数并将数据喂给识别器训练
    print('Training...')
    faces,ids = get_images_and_labels(path,detector)
    #训练模型
    recog.train(faces,np.array(ids))
    #保存模型(把里面原有的数据清除，然后再往里面写数据)
    recog.save('trainner/trainner_gg.yml')
    print('Training complete')

#每次录入完新的人脸数据后，更新 ’my_config‘ 中的总人数和添加新的用户名0
def rewrite_my_config():
    f = open('my_config.txt', 'r+')
    Tatal_face_num=f.readline()
    f.close()

    #把新的人脸信息录入到’my_config‘中
    f=open('my_config.txt','a')
    new_face_id=str(int(Tatal_face_num))
    f.write(new_face_id+" "+'User'+new_face_id+'\n')
    f.close()

    #把总人脸数加一
    f = open('my_config.txt', 'r+')
    flist = f.readlines()
    flist[0] = str(int(flist[0]) + 1) + " \n"
    f.close()

    #把修改后的数据再存入’my_config‘中
    f = open('my_config.txt', 'w+')
    f.writelines(flist)
    f.close()

if __name__=='__main__':
    # init()
    get_face()
    # finish_promote()
    train_face()
    rewrite_my_config()