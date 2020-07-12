import json
from pycocotools.coco import COCO
from pycocotools.mask import *
import numpy as np
import os
del os.sys.path[1]
import cv2
import yaml

imagepath = '/home/huo/DL/pose-interpreter-networks/data/image'
totalfilepath = '/home/huo/DL/pose-interpreter-networks/data/renameimages'
#function to get particular file list
def get_file_list(file_dir, file_tail, sep_dir):
	file_list = []
	all_file = os.listdir(file_dir)
	for files in all_file:
		if os.path.splitext(files)[0][-4:] == file_tail:
			file_list.append(files)
			#olddir = os.path.join(file_dir, files)
			#shutil.copy(olddir, os.path.join(sep_dir, files))
	return file_list

result = {}
ann_gt = []
ann_dic = {}
#result["info"] = {"description":None,"version":1.0,"year":2018,"contributor":"Huo","date_created":1544011502.001342}
#result["camera"] = {"name":"kinect1","K":[525,0.0,319.5,0.0,525,239.5,0.0,0.0,1.0],"height":480,"width":640,"camera_type":"kinect2","id":0}
label = '0000000001_labels.png'
imlab = cv2.imread(os.path.join(totalfilepath, label), cv2.IMREAD_GRAYSCALE)
ret, imgbw = cv2.threshold(imlab,0,255,cv2.THRESH_BINARY)
# order='F' convrt the default format in row major order to column major order
imgbw = np.asarray(imgbw[:,:],order='F')
rle = encode(imgbw)
ar = int(area(rle))
box = toBbox(rle)
box = box.tolist()
rle ["counts"] = str(rle["counts"], encoding='utf-8')
ann_dic["segmentation"] = rle
ann_dic["area"] = ar
poses = {}
yamls = "0000000001_poses.yaml"
with open (os.path.join(totalfilepath, yamls)) as y:
	pose_para = y.read()
	pose_dic = yaml.load(pose_para)

pose_value = pose_dic['charge_pile']
pose_list = pose_value['pose']
x = pose_list[0][0]
y = pose_list[0][1]
z = pose_list[0][2]
q0 = pose_list[1][0]
qx = pose_list[1][1]
qy = pose_list[1][2]
qz = pose_list[1][3]
positions = {}
positions['x'] = x
positions['y'] = y
positions['z'] = z
orientations = {}
orientations['x'] = qx
orientations['y'] = qy
orientations['z'] = qz
orientations['w'] = q0
poses['position'] = positions
poses['orientation'] = orientations
ann_dic["pose"] = poses
ann_dic["iscrowd"] = 0
ann_dic["image_id"] = 1
ann_dic["bbox"] = box
ann_dic["category_id"] = 1
ann_dic["id"] = 1
ann_gt.append(ann_dic)

result = ann_gt
##read into dic
#pose = "000000"+ item[6: 10]+ "_poses.yaml"

"""
img = cv2.imread(os.path.join(totalfilepath, '0000000001_labels.png'),cv2.IMREAD_GRAYSCALE)
print(img.shape)
ret, imgbw = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
print(imgbw.shape)
#cv2.imshow('bw',imgbw)
#cv2.waitKey(0)
imgbw = np.asarray(imgbw[:,:],order='F')   ##convrt the default format in row major order to column major order
rle = encode(imgbw)
#R = merge(rle, intersect=False)
ar = area(rle)
box = toBbox(rle)
print(rle)
#print(R)
print(ar)
print(box)
"""

#result["annotations"] = ann_gt
#result["licenses"] = {"url":"https://opensource.org/licenses/BSD-3-Clause","id":0,"name":"BSD-3-Clause"}
filename='test.json'
with open(filename,'w') as ong:
    json.dump(result,ong)