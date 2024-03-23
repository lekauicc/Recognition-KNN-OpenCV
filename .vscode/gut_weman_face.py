from PIL import Image
import os
imageface=r'C:\Users\MI\Desktop\face_dis\face\heiren'
targeface=r'C:\Users\MI\Desktop\face_dis\face\not_heiren'
count=6
for face in os.listdir(imageface):
    imagepath=os.path.join(imageface,face)
    sign=os.path.split(imagepath)[-1].split('.')[0:1]
    if sign[0]=='MK':
        if face.endswith('tiff'):
            im=Image.open(imagepath)
            if im.mode == "P":
                im = im.convert('RGB')
            targetname=os.path.join(targeface,str(count)+face[2:-5]+'.jpg')
            # count+=1
            im.save(targetname)