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
    #  classifier = train(r"Myfiles\train_face", model_save_path=r"Myfiles\model\trained_knn_model.clf", n_neighbors=2)
# def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
def train(train_dir, model_save_path, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
        train_dir ; 给定的训练目录,格式是给定的目录下有多个目录,这多个目录以人的名字命名,每个目录下有一张或者多张想要训练的人的面部图片,一张图片有且只有一张脸,不然不会识别
        model_save_path :  训练模型保存的路径
        n_neighbors ; 分类中要加权的邻居数,不选择的话,就默认
        knn_algo : 要支持的基础数据结构knn.默认值是树
        verbose : 训练的详细程度
    """
    X = []
    y = []
    # 对训练集中的每个人进行循环
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        # 循环浏览当前人员的每个训练图像
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # 如果训练图像中没有人（或人太多），请跳过该图像
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # 将当前图像的人脸编码添加到训练集中
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
    # 确定在KNN分类器中用于加权的邻居数
    if n_neighbors is None:

        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    # 创建并训练KNN分类器
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # 保存经过训练的KNN分类器
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf



# predictions = predict(rgb_small_frame, model_path=r"Myfiles\model\trained_knn_model.clf") 
def predict(rgb_small_frame, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    使用训练好的KNN分类器识别给定图像中的人脸
    X_img_path: 要识别的图像
    knn_clf: （可选）knn分类器对象。如果未指定，则必须指定型号\保存\路径。
    model_path: （可选）到酸洗knn分类器的路径。如果未指定，则模型\保存\路径必须为knn\ clf。
    distance_threshold: （可选）人脸分类的距离阈值。它越大，机会就越大,把一个不认识的人误分类为一个已知的人
    :return: 图像中已识别面的名称和面位置列表：[（名称，边界框），…]。对于未识别的人脸，将返回名称“未知”
    """
 
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    # 加载图像文件并查找人脸位置坐标
    X_face_locations = face_recognition.face_locations(rgb_small_frame)
 
    # 如果在图像中找不到面，则返回空结果
    if len(X_face_locations) == 0:
        return []
    # 在测试图像中查找面编码
    faces_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=X_face_locations)
    # 使用KNN模型找到测试面的最佳匹配
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    #判断找到的这个最佳匹配的人能否，，满足匹配度distance_threshold的要求
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print(closest_distances[0][0][0])
    # 预测类并删除不在阈值内的分类
    # y_pred=knn_clf.predict(faces_encodings)
    # knn_clf.
    # accuracy= accuracy_score(y_pred,)
    answer=[(pred, loc) if rec else ("unknow", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    return answer
 
 
if __name__ == '__main__':
    # print("Training KNN classifier...")
    # # 训练完后,注释掉即可,没有必要重新生成
    # classifier = train(r"Myfiles\train_face", model_save_path=r"Myfiles\model\trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")

    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    while True:
        # 抓取一帧视频
        ret, frame = video_capture.read()
        # 将视频帧调整为1/4大小以加快人脸识别处理
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
        rgb_small_frame = small_frame[:, :, ::-1]
        # 只处理每隔一帧的视频以节省时间
        if process_this_frame:
            predictions = predict(rgb_small_frame, model_path=r"Myfiles\model\trained_knn_model.clf")
        process_this_frame = not process_this_frame
        # 显示结果
        for name, (top, right, bottom, left) in predictions:
            # 缩放面部位置，因为我们在中检测到的帧已缩放为1/4大小
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # 在脸上画一个方框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # cv2不支持中文所以先转换为pil,添加文字,然后再转换回去
            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 图像从OpenCV格式转换成PIL格式
            draw = ImageDraw.Draw(img_PIL)
            draw.text((left + 6, bottom - 35), name, font=ImageFont.truetype("simsun.ttc", 30, encoding="unic"))
            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式
        # 显示结果图像
        cv2.imshow('Video', frame)
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头的资源
    video_capture.release()
    cv2.destroyAllWindows()