'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-05-14 14:25:57
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-05-25 16:49:31
FilePath: \face_dis\.vscode\get_face_by_cam.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#-----获取人脸样本-----
import cv2
 
#调用视频流
cap = cv2.VideoCapture("http://192.168.137.235:8081//stream")
# cap = cv2.VideoCapture(0)
#调用人脸分类器
face_detector = cv2.CascadeClassifier(r'haarcascades\haarcascade_frontalface_alt2.xml')  
# face_detector = cv2.CascadeClassifier(r'haarcascade_frontalface_alt2.xml')  
#为即将录入的脸标记一个id
face_id = input('\n User data input,Look at the camera and wait ...')
#sampleNum用来计数样本数目
count = 0
 #调用笔记本内置摄像头，参数为0，如果有其他的摄像头可以调整参数为1,2
while True:    
    '''从摄像头读取图片.其中success是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，
    它的返回值就为False。img就是每一帧的图像，是个三维矩阵。'''
    success,img = cap.read()    
    #转为灰度图片，减少程序符合，提高识别度
    if success is True: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    else:   
        break
    #检测人脸，将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸
    '''detectMultiScale采用滑动窗口的方式选取人脸
       其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
        scaleFactor越大，检测的速度越快，但正确率可能有所下降；
        minNeighbors=5，代表迭代后检测到的一个人的人脸数目要大于5才会保存他的人脸数据
    '''
    cv2.imshow('image',img)  
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4)
 
    #框选人脸，for循环保证一个能检测的实时动态视频流
    for (x, y, w, h) in faces:

        '''xy为左上角的坐标,w为宽，h为高，(x+w, y+w)是右下角的坐标(y轴正半轴是向下的)。用rectangle为人脸标记画框(y)
        
        '''
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))#采集一次人脸数据
        #成功框选则样本数增加
        count += 1  
        #保存图像，把灰度图片看成二维数组来检测人脸区域
        #(这里是建立了data的文件夹，当然也可以设置为其他路径或者调用数据库)
        cv2.imwrite(r"face\low_dis_face\ "+str(face_id)+'.'+str(count)+'.jpg',gray[y:y+h,x:x+w]) 
        #显示图片
        cv2.imshow('image',img)       
    #保持画面的连续。waitkey方法可以绑定按键保证画面的收放，通过q键退出摄像
    k = cv2.waitKey(1) #在1ms内，按下‘s’键，退出程序（ord(s)==27）    
    if k == '27':
        break        
        #或者得到10个样本后退出摄像，这里可以根据实际情况修改数据量(实际测试后800张的效果是比较理想的)
    elif count >= 500:
        break
 
#关闭摄像头，释放资源
# cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
# 对于电脑摄像头
'''
人脸检测时不能乱动
戴眼镜会严重影响人脸检测的精度
周围的亮度会影响人脸检测的效果
haarcascade_frontalface_default.xml 对戴眼镜的采集效果还能接收
haarcascade_frontalface_alt2.xml的效果————戴眼镜时要与摄像头保持一定距离才能检测到人脸，对戴眼镜的采集效果还能接收
'''
# 对于URL
'''
戴眼镜会严重影响人脸检测的精度，要靠的比较近时才能框出人脸
周围的亮度会影响人脸检测的效果

'''