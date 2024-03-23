from socket import *
import cv2
import io
from PIL import Image
import numpy as np
import os
import identify_face_last
import entry_face_last
import sqlite3

def creat_table():
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

server_s = socket(AF_INET, SOCK_STREAM)
# 2. 绑定本地信息
server_s.bind(("", 7710))

# 3. 设置为被动的
server_s.listen(128)
# 4. 等待客户端链接
new_s, client_info = server_s.accept()
while(1):
    # 5. 用新的套接字为已经连接好的客户端服务器
    recv_content = new_s.recv(1024)
    print(recv_content.decode('utf-8'))
    if(recv_content.decode('utf-8')=='0'):
        print('dd')
        entry_face_last.get_face()
        entry_face_last.finish_promote()
        entry_face_last.train_face()
        entry_face_last.write_my_config()
    if(recv_content.decode('utf-8')=='1'):
        print('ss')
        identify_face_last.init()
        identify_face_last.get_face()
        identify_face_last.identify_face()
        identify_face_last.dellete_pic()

if __name__=='__main__':
    creat_table()