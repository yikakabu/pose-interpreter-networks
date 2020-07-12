"""
use this file to make a single_object dataset of ChargePile
2019/04/16 wrote by Huo
"""

import os
import json
import scipy.io as sio
import numpy as np
import shutil
from pycocotools.coco import COCO
from pycocotools.mask import *
del os.sys.path[1]
import cv2
import yaml
import random

sourcefilepath1 = '/home/huo/DL/TestData/logs/2018-12-05-00/images'
sourcefilepath2 = '/home/huo/DL/TestData/logs/2018-12-05-01/images'
totalfilepath = '/home/huo/DL/pose-interpreter-networks/data/renameimages'
imagepath = '/home/huo/DL/pose-interpreter-networks/data/ChargePileDataset/image'
depthpath = '/home/huo/DL/pose-interpreter-networks/data/ChargePileDataset/depth'
train_percent = 0.8

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

#function to remove color label file list
def remove_color_list(file_dir):
	file_list = []
	all_file = os.listdir(file_dir)
	for files in all_file:
		if os.path.splitext(files)[0][-8:] == 'r_labels':
			os.remove(os.path.join(file_dir,files))
	#return file_list

#function to rename file(add 4324 to the filename of the second filefolder)
def renameall(file_dir, target_dir):
	#shutil.copytree(file_dir,target_dir) 
	all_file = os.listdir(file_dir)
	for files in all_file:
		olddir = os.path.join(file_dir, files)
		filename = os.path.splitext(files)[0]
		filetype = os.path.splitext(files)[1] 
		newdir = os.path.join(target_dir, '000000'+ str(int(filename[:10])+4324)+ filename[10:]+ filetype)
		shutil.copy(olddir, newdir)

def json_image_ann(images_set):
	img_gt = []
	ann_gt = []
	i=0
	for item in images_set:
		im_dic = {}
		ann_dic = {}
		img = cv2.imread(os.path.join(imagepath, item))
		#height, width, _ = img.shape
		im_dic["license"] = 0
		im_dic["file_name"] = item
		im_dic["coco_url"] = ""
		im_dic["height"] = img.shape[0]
		im_dic["width"] = img.shape[1]
		utime="000000"+ item[6: 10]+ "_utime.txt"
		with open (os.path.join(totalfilepath, utime)) as u:
			time = u.read()
			time = int(time[:10])
		im_dic["date_captured"] = float('%.1f' % time)
		im_dic["camera_id"] = 0
		im_dic["flickr_url"] = ""
		im_dic["id"] = i
		img_gt.append(im_dic)
		# annotations
		label = "000000"+ item[6: 10]+ "_labels.png"
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
		yamls = "000000"+ item[6: 10]+ "_poses.yaml"
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
		##read into dic
		ann_dic["pose"] = poses
		ann_dic["iscrowd"] = 0
		ann_dic["image_id"] = i
		ann_dic["bbox"] = box
		ann_dic["category_id"] = 1
		n = i    #single object n=i
		ann_dic["id"] = n
		ann_gt.append(ann_dic)
		i+=1
	return img_gt, ann_gt


#function to dump json
def dump_json(image_set):
	result = {}
	result["info"] = {"description":None,"version":1.0,"year":2018,"contributor":"Huo","date_created":1544011502.001342}
	result["cameras"] = [{"name":"kinect1","K":[525,0.0,319.5,0.0,525,239.5,0.0,0.0,1.0],"height":480,"width":640,"camera_type":"kinect1","id":0}]
	result["licenses"] = [{"url":"https://opensource.org/licenses/BSD-3-Clause","id":0,"name":"BSD-3-Clause"}]
	result["images"] ,result["annotations"]= json_image_ann(image_set)
	result["categories"] = [{"supercategory":"objects","mesh":"charge_pile.stl","id":1,"name":"charge_pile"}]
	#print(result)
	return result

if __name__ == "__main__":

	#labellist
	#shutil.copytree(sourcefilepath1, totalfilepath) 
	#renameall(sourcefilepath2, totalfilepath)
	#remove_color_list(totalfilepath)
	#rgblist = get_file_list(totalfilepath, '_rgb', imagepath)
	#depthlist = get_file_list(totalfilepath, 'epth', depthpath)
	
	#print(rgblist)
	#print(len(rgblist))
	images = os.listdir(imagepath)
	num = len(images)
	train_images = random.sample(images, int(train_percent*num))
	test_images = [item for item in images if item not in train_images]
	filename='20190323_ChargePile.json'
	result = dump_json(images)
	with open(filename,'w') as ong:
		json.dump(result,ong)
	filename='train_20190323_ChargePile.json'
	result = dump_json(train_images)
	with open(filename,'w') as ong:
		json.dump(result,ong)
	filename='val_20190323_ChargePile.json'
	result = dump_json(test_images)
	with open(filename,'w') as ong:
		json.dump(result,ong)
