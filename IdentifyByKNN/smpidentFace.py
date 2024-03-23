'''
该死的Raspberry pi Camera像素太辣鸡了，使用face_recognition.face_encodings函数时根本检测不到人脸，
所以只能先把拍摄照片保存下来，利用照片来做人脸识别匹配，识别完成后再把照片删完，用于下次人脸识别
'''
from socket import *
import cv2
import io
from PIL import Image
import numpy as np
import os
import time
import numpy as np
import mysql.connector as MysqlCon
import math
import os
import pickle
import re
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from face_recognition.face_detection_cli import image_files_in_folder
from sklearn import neighbors
from sklearn.metrics import accuracy_score

import OpenDoor
import FaceDatabase


def get_faceInf():
    # 连接到SQLite数据库
    conn = MysqlCon.connect(
        host="localhost", user="remote_user", password="las", database="AccessControl"
    )
    # 创建游标对象
    cursor = conn.cursor()
    try:
        Id_dict = []
        # 执行查询语句
        cursor.execute("SELECT * FROm FaceInf")
        # 获取所有数据
        rows = cursor.fetchall()
        # 打印每一行数据
        for row in rows:
            Id_dict.append(row[1])
            # print(row)
        # print(Id_dict)
        return Id_dict
    except Exception as e:
        print(e)
        print("获取人员名字错误")
        # conn.rollback()
    finally:
        # 关闭连接
        conn.close()


# def idendify_faces():
#     #调用视频流
#     cap = cv2.VideoCapture(0)
#     #调用人脸分类器
#     face_detector = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
#     #识别方法
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     #count用来计数样本数目
#     count = 0
#     #secc_num用于统计识别成功的次数
#     secc_num=0
#     #使用之前训练好的模型
#     recognizer.read(r'FaceModle/trainner.yml')

#     #再次调用人脸分类器
#     cascade_path = "haarcascade_frontalface_default.xml"
#     face_detector = cv2.CascadeClassifier(cascade_path)

#     #加载一个字体，用于识别后，在图片上标注出对象的名字
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     cap.set(3,480)               #cap.set 摄像头参数设置
#     cap.set(4,480)               #3代表图像高度，4代表图像宽度，5代表图像帧率
#     cap.set(5,40)                #图像高为600，宽度为480，帧率为40

#     # id_dict=["User1","User2","User3","User4","Host"]
#     #获取录入的所有人的姓名
#     id_dict=get_faceInf()
#     #recog_id[]会保存每次预测成功的人的ID
#     recog_id=[]

#     while True:
#         '''从摄像头读取图片.其中success是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，
#             它的返回值就为False。img就是每一帧的图像，是个三维矩阵。'''
#         success,img = cap.read()
#         #转为灰度图片，减少程序符合，提高识别度
#         if success is True:
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         else:
#             break
#         #检测人脸，将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸
#         '''detectMultiScale采用滑动窗口的方式选取人脸
#         其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
#             scaleFactor越大，检测的速度越快，但正确率可能有所下降；
#             minNeighbors=5，代表迭代后检测到的一个人的人脸数目要大于5才会保存他的人脸数据
#         '''
#         cv2.imshow('image',img)
#         faces = face_detector.detectMultiScale(gray, scaleFactor=1.6 ,
#                                                minNeighbors=4)
#         #框选人脸，for循环保证一个能检测的实时动态视频流
#         for (x, y, w, h) in faces:

#             '''xy为左上角的坐标,w为宽，h为高，(x+w, y+w)是右下角的坐标(y轴正半轴是向下的)。用rectangle为人脸标记画框(y)

#             '''
#             cv2.rectangle(img, (x, y), (x+w, y+w), (0, 255, 0),5)#采集一次人脸数据
#             # cv2.imwrite(r"CeshiFace\ "+str(1)+'.'+str(count)+'.jpg',gray[y:y+h,x:x+w])
#             idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])# ldnum表示识别出的人脸对应与之前训练好后得到的 'trainner/trainner_gg.yml' 中的id，(100-confidence) 表示识别对用的概率

#             #计算出一个检验结果
#             if confidence < 85:
#                 idum = id_dict[idnum]
#                 confidence = "{0}%",format(round(100-confidence))
#                 print(count,secc_num,idum,confidence)
#                 recog_id.append(idnum)
#                 secc_num+=1
#             else:
#                 idum = "unknown"
#                 confidence = "{0}%",format(round(100-confidence))
#                 print(idum,confidence)
#             #成功框选则样本数增加
#             count += 1
#             # 显示检验结果以及用户名
#             # cv2.putText(img,idum,(x+5,y-5),font,1,(0,0,255),1)
#             # cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(0,0,0),1)
#             cv2.putText(img,"HY",(x+5,y-5),font,1,(0,0,255),2)
#             cv2.putText(img,str("63%"),(x+5,y+h-5),font,1,(0,0,0),2)


#             # #将人脸框出来
#             cv2.imshow('image',img)


#         #录入人脸时按下“q”，强制退出录入·
#         if cv2.waitKey(1) == ord("q"):
#             break
#         elif count >= 100:
#             break
#     #关闭摄像头，释放资源
#     # cv2.destroyAllWindows()
#     cap.release()
#     cv2.destroyAllWindows()
#     #判断识别是否为内部人员，若为，则执行开门程序
#     try:
#         if (secc_num/count)>0.80:
#             #开门
#             OpenDoor.open_door()
#             #找出匹配度最高的人员
#             result = max(set(recog_id), key=recog_id.count)
#             #向AccessedInf表中插入数据
#             FaceDatabase.Insert_AccessedInf(id_dict[result],'人脸识别开门')
#     except Exception as e:
#         print(e)


def get_face():
    # 调用视频流
    cap = cv2.VideoCapture(0)
    # 调用人脸分类器
    face_detector = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
    # 为即将录入的脸标记一个id
    face_id = 1
    # sampleNum用来计数样本数目
    count = 0

    cap.set(3, 600)  # cap.set 摄像头参数设置
    cap.set(4, 480)  # 3代表图像高度，4代表图像宽度，5代表图像帧率
    cap.set(5, 40)  # 图像高为600，宽度为480，帧率为40

    while True:
        """从摄像头读取图片.其中success是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，
        它的返回值就为False。img就是每一帧的图像，是个三维矩阵。"""
        success, img = cap.read()
        # 转为灰度图片，减少程序符合，提高识别度
        if success is True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break
        # 检测人脸，将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸
        """detectMultiScale采用滑动窗口的方式选取人脸
        其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
        scaleFactor越大，检测的速度越快，但正确率可能有所下降；
        minNeighbors=5，代表迭代后检测到的一个人的人脸数目要大于5才会保存他的人脸数据
        """
        cv2.imshow("image", img)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.6, minNeighbors=4)
        # 框选人脸，for循环保证一个能检测的实时动态视频流
        for x, y, w, h in faces:
            """xy为左上角的坐标,w为宽，h为高，(x+w, y+w)是右下角的坐标(y轴正半轴是向下的)。用rectangle为人脸标记画框(y)"""
            cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 3)  # 采集一次人脸数据
            # 成功框选则样本数增加
            count += 1
            # 保存图像，把灰度图片看成二维数组来检测人脸区域
            # (这里是建立了data的文件夹，当然也可以设置为其他路径或者调用数据库)
            cv2.imwrite(
                r"TemporaryFace/" + str(face_id) + "." + str(count) + ".jpg",
                gray[y : y + h, x : x + w],
            )
            # 将人脸框出来
            cv2.imshow("image", img)
        # 录入人脸时按下“q”，强制退出录入·
        # if cv2.waitKey(1) == ord("q"):
        #     break
        # 每次录入30张照片,l录入完成后向人脸数据插入新的人脸数据
        if count >= 10:
            print("录入完成")
            break
#删除临时人脸照片
def dellete_pic():
    path=r'Temporaryface'
    image_paths=[os.path.join(path,f) for f in os.listdir(path)]
    for image_path in image_paths:
        os.remove(image_path)

def predict(rgb_small_frame, model_path):
    """
    使用训练好的KNN分类器识别给定图像中的人脸
    X_img_path: 要识别的图像
    knn_clf: （可选）knn分类器对象。如果未指定，则必须指定型号\保存\路径。
    model_path: （可选）到酸洗knn分类器的路径。如果未指定，则模型\保存\路径必须为knn\ clf。
    distance_threshold: （可选）人脸分类的距离阈值。它越大，机会就越大,把一个不认识的人误分类为一个已知的人
    :return: 图像中已识别面的名称和面位置列表：[（名称，边界框），…]。对于未识别的人脸，将返回名称“未知”
    """
    # 参数设置
    knn_clf = None
    distance_threshold = 0.6
    if knn_clf is None and model_path is None:
        raise Exception(
            "Must supply knn classifier either thourgh knn_clf or model_path"
        )
    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, "rb") as f:
            knn_clf = pickle.load(f)
    # 加载图像文件并查找人脸位置坐标
    X_face_locations = face_recognition.face_locations(
        rgb_small_frame, number_of_times_to_upsample=3
    )
    # 如果在图像中找不到面，则返回空结果
    if len(X_face_locations) == 0:
        return []
    #通过人脸位置坐标把人脸数据取出来
    faces_encodings = face_recognition.face_encodings(
        rgb_small_frame, known_face_locations=X_face_locations
    )
    # 使用KNN模型找到测试面的最佳匹配
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    # 判断找到的这个最佳匹配的人能否，，满足匹配度distance_threshold的要求
    are_matches = [
        closest_distances[0][i][0] <= distance_threshold
        for i in range(len(X_face_locations))
    ]
    # 暂时以距离作为可信度
    print(closest_distances[0][0][0])
    match = str(closest_distances[0][0][0])
    if are_matches and match>0.9:
        answer=[(knn_clf.predict(faces_encodings),match)]
    else:
        print("不认识啊？？")
    return answer



def identify_faces_knn():
    # 获取录入的所有人的姓名
    id_dict = get_faceInf()
    # 临时人脸存放目录
    train_dir = r"Temporaryface"
    # 遍历每张照片，如果识别成功，发送开门指令
    image_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    for path in image_paths:
        frame = cv2.imread(path)
        rgb_small_frame = frame[:, :, ::-1]
        # 只处理每隔一帧的视频以节省时间
        predictions = predict(
            rgb_small_frame, model_path=r"Modei\trained_knn_model.clf"
        )
        for ids, confi in predictions:

            #通过ids获取匹配成功的人员消息
            id=int(ids[0])
            name=id_dict[id]
            # 如果匹配度在90%以上，则视为识别成功
            try:
                if (confi) > 0.90:
                    print("识别成功，匹配结果为："+name+" 识别准确的为："+confi)
                    # 开门
                    OpenDoor.open_door()
                    # 向AccessedInf表中插入数据
                    FaceDatabase.Insert_AccessedInf(name, "人脸识别开门")
                    #删除临时人脸文件夹中的数据
                    dellete_pic()
                    break
            except Exception as e:
                print(e)
    #就算没有识别成功，也要删除临时人脸文件夹中的数据
    dellete_pic()


if __name__ == "__main__":
    identify_faces_knn()
    # idendify_faces()
