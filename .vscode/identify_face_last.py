from socket import *
import cv2
import io
from PIL import Image
import numpy as np
import os
import sqlite3
import time
import numpy as np
id_dict={}
Tatal_face_num=1


def creat():
    # 1.硬盘上创建连接
    con = sqlite3.connect(r'C:\Users\MI\Desktop\face_dis\first1.db')
    # 获取cursor对象
    cur = con.cursor()
    # 执行sql创建表
    sql = 'create table t_person(pno INTEGER PRIMARY KEY  AUTOINCREMENT  ,Name varchar(30) NOT NULL,Time varchar(30) NOT NULL)'
    try:
        cur.execute(sql)
        print('创建成功')
    except Exception as e:
        print(e)
        print('创建表失败')
    finally:
        # 关闭游标
        cur.close()
        # 关闭连接
        con.close()

def insert(data):
    con = sqlite3.connect(r'C:\Users\MI\Desktop\face_dis\first1.db')
    cur=con.cursor()
    sql='insert into t_person(Name,Time) values(?,?)'
    try:
        
        cur.execute(sql,data)
        con.commit()
        print('数据添加成功')
    except Exception as e:
        print(e)
        print('添加错误')
        con.rollback()
    finally:
        cur.close()
        con.close()


def init():  # 将config文件内的信息读入到字典中
    f = open(r'my_config.txt')
    global Total_face_num
    Total_face_num = int(f.readline())
    for i in range(int(Total_face_num)):
        line = f.readline()
        id_name = line.split(' ')
        id_dict[int(id_name[0])] = id_name[1]
    f.close()


def get_face():
    #调用人脸分拣器
    face_detector = cv2.CascadeClassifier(r'C:\Users\MI\Desktop\face_dis\haarcascade_frontalface_default.xml')  
    #创建UDP套接字。若使用TCP，掉了一帧就要重新发送，延迟很大，而且会导致cam端结束拍摄
    s = socket(AF_INET, SOCK_DGRAM, 0)
    s.bind(("0.0.0.0", 8080))
    #照片计数变量
    count=0
    #为即将录入的脸标记一个id
    f=open('my_config.txt','r')
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
            cv2.imwrite(r"C:\Users\MI\Desktop\face_dis\face\identified_face\ "+str(face_id)+'.'+str(count)+'.jpg',gray[y:y+h,x:x+w]) 
            #将人脸框出来
            cv2.imshow('image',img)   
        if cv2.waitKey(1) == ord("q"):
            break
        elif count >= 100:
                break

def open_door():
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)

    # 2. 链接服务器
    tcp_client_socket.connect(("192.168.43.118", 7788))

    # 提示用户输入数据
    send_data ='1'

    # 3. 向服务器发送数据
    tcp_client_socket.send(send_data.encode("utf-8"))

    # 4. 关闭套接字
    tcp_client_socket.close()




def identify_face():
    pass_face=[]
    #识别方法
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    #使用之前训练好的模型
    recognizer.read('trainner/trainner_gg.yml')
    
    #再次调用人脸分类器
    cascade_path = "haarcascade_frontalface_default.xml" 
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    #加载一个字体，用于识别后，在图片上标注出对象的名字
    # font = cv2.FONT_HERSHEY_SIMPLEX
    
    idnum = 0
    #设置好与ID号码对应的用户名，如下，如0对应的就是初始
    
    #这里只是为了使用笔记本摄像头默认的边框大小
    cam = cv2.VideoCapture(0)

    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    path=r'C:\Users\MI\Desktop\face_dis\face\identified_face'
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]
    count=0
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #识别人脸
        faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 1,
                minSize = (int(minW),int(minH))
                )
        #进行校验
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])# ldnum表示识别出的人脸对应与之前训练好后得到的 'trainner/trainner_gg.yml' 中的id，(100-confidence) 表示识别对用的概率
            pass_face.append(idnum)
            #计算出一个检验结果
            if confidence < 85:
                idum = id_dict[idnum]
                confidence = "{0}%",format(round(100-confidence))
                print(count,idnum,idum,confidence)
                count+=1
            else:
                idum = "unknown"
                confidence = "{0}%",format(round(100-confidence))
                print(idum,confidence)
   
    #输出识别成功的人数 
    '''
    这里可以通过判断识别成功的次数来决定是否开门
    '''
    print('识别成功率为{}%'.format(count))
    if count>75:
        result = max(set(pass_face), key=pass_face.count)
        # 识别成功后向 first1.db 数据库中加入识别成功的人的信息和识别时间
        insert([id_dict[result],time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())])
        open_door()
        #输出检验结果以及用户名
        # cv2.putText(img,str(idum),(x+5,y-5),font,1,(0,0,255),1)
        # cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(0,0,0),1)

        #展示结果
        # cv2.imshow('camera',img)
        # k = cv2.waitKey(20)
        # if k == 27:
        #     break
    
    #释放资源
    cam.release()
    cv2.destroyAllWindows()

#每次识别人脸后，将identified_face里的照片删除，方便下次识别新的人脸
def dellete_pic():
    path=r'face\identified_face'
    image_paths=[os.path.join(path,f) for f in os.listdir(path)]
    for image_path in image_paths:
        os.remove(image_path)


if __name__=='__main__':
#     creat()
    init()
    get_face()
    identify_face()
    dellete_pic()


