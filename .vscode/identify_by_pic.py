'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2022-11-08 19:35:20
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-05-14 15:38:30
FilePath: \face_dis\.vscode\identify_by_pic.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#-----检测、校验并输出结果-----
import cv2
import os  
#准备好识别方法
recognizer = cv2.face.LBPHFaceRecognizer_create()
 
#使用之前训练好的模型
recognizer.read(r'trainner\trainner.yml')
 
#再次调用人脸分类器
cascade_path = "haarcascade_frontalface_default.xml" 
face_cascade = cv2.CascadeClassifier(cascade_path)
 
# 加载一个字体，用于识别后，在图片上标注出对象的名字
font = cv2.FONT_HERSHEY_SIMPLEX
 
idnum = 0
#设置好与ID号码对应的用户名，如下，如0对应的就是初始
names = ['myself','user1','user2','user3','user4','user5','user6']
 
#调用摄像头
cam = cv2.VideoCapture(0)

minW = 0.1*cam.get(3)#64.0
minH = 0.1*cam.get(4)#48.0
path=r'face\getfaceby_comp'
image_paths = [os.path.join(path,f) for f in os.listdir(path)]
i=0
for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #识别并获取人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2)
    #进行校验
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])

        #计算出一个检验结果
        if confidence < 85:
            idum = names[idnum]
            confidence = "{0}%",format(round(100-confidence))
            print(i,idum,confidence)
            i+=1
        else:
            idum = "unknown"
            confidence = "{0}%",format(round(100-confidence))
            print(idum,confidence)
    #假设每张图片的人脸都能被框出来
    try:
        # 输出检验结果以及用户名
        cv2.putText(img,"user1",(x+5,y-5),font,1,(0,0,255),1)
        cv2.putText(img,"87%",(x+5,y+h-5),font,1,(0,0,0),1)
    except:
        print("没有检测到人脸")

    # 展示结果
    cv2.imshow('camera',img)
    k = cv2.waitKey(20)
    if k == 27:
        break
#打印出识别成功的次数
print(i)
#释放资源
cam.release()
cv2.destroyAllWindows()