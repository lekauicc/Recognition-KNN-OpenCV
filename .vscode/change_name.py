'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2022-11-22 17:24:09
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-05-23 20:55:41
FilePath: \face_dis\.vscode\change_name.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
path=r'face\wemen_face'
image_paths=[os.path.join(path,f) for f in os.listdir(path)]
for image_path in image_paths:
    if os.path.split(image_path)[-1].split('.')[0]=='1':
        new_image_path=os.path.join(os.path.split(image_path)[0],"0"+'.'+os.path.split(image_path)[-1].split('.')[1]+'.'+'jpg')
    if os.path.split(image_path)[-1].split('.')[0]=='3':
        new_image_path=os.path.join(os.path.split(image_path)[0],"1"+'.'+os.path.split(image_path)[-1].split('.')[1]+'.'+'jpg')
    if os.path.split(image_path)[-1].split('.')[0]=='4':
        new_image_path=os.path.join(os.path.split(image_path)[0],"2"+'.'+os.path.split(image_path)[-1].split('.')[1]+'.'+'jpg')
    if os.path.split(image_path)[-1].split('.')[0]=='6':
        new_image_path=os.path.join(os.path.split(image_path)[0],"3"+'.'+os.path.split(image_path)[-1].split('.')[1]+'.'+'jpg')

    os.rename(image_path,new_image_path)