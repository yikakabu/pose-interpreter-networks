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

sourcefilepath1 = '/home/huo/DL/TestData/logs/2019-05-16-01/images'
sourcefilepath2 = '/home/huo/DL/TestData/logs/2019-05-16-00/images'
totalfilepath = '/home/huo/DL/pose-interpreter-networks/data/total'
imagepath = '/home/huo/DL/pose-interpreter-networks/data/CokeDataset/image'
depthpath = '/home/huo/DL/pose-interpreter-networks/data/CokeDataset/depth'
train_percent = 0.8

#function to get particular file list
def get_file_list(file_dir, file_tail, sep_dir):
	file_list = []
	all_file = os.listdir(file_dir)
	for files in all_file:
		if os.path.splitext(files)[0][-4:] == file_tail:
			file_list.append(files)
			olddir = os.path.join(file_dir, files)
			shutil.copy(olddir, os.path.join(sep_dir, files))
	return file_list

#function to remove color label file list
def remove_color_label(file_dir):
	file_list = []
	all_file = os.listdir(file_dir)
	for files in all_file:
		if os.path.splitext(files)[0][-8:] == 'r_labels':
			os.remove(os.path.join(file_dir,files))
	#return file_list

#function to rename file(add 4324 to the filename of the second filefolder)
def renameall(file_dir, target_dir):
	all_file = os.listdir(file_dir)
	addnum = 29255 // 5    #the amount of file in the first folder
	for files in all_file:
		olddir = os.path.join(file_dir, files)
		filename = os.path.splitext(files)[0]
		filetype = os.path.splitext(files)[1] 
		newdir = os.path.join(target_dir, '000000'+ str(int(filename[:10]) + addnum) + filename[10:] + filetype)
		shutil.copy(olddir, newdir)

def json_image_ann(images_set):
	img_gt = []
	ann_gt = []
	i = 0   # images index
	j = 0   # masks index
	for item in images_set:
		im_dic = {}
		img = cv2.imread(os.path.join(imagepath, item))
		#height, width, _ = img.shaped
		im_dic["license"] = 0
		im_dic["file_name"] = item
		im_dic["coco_url"] = ""
		im_dic["height"] = img.shape[0]
		im_dic["width"] = img.shape[1]
		utime = "000000" + item[6: 10] + "_utime.txt"
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
		#label = '/home/huo/DL/LabelFusion_Sample_Data/logs/2017-06-15-73/images/0000000002_labels.png'
		lb_list = [1, 2, 3, 4]
		obj_list = ['kinect', 'bottle', 'cola_can', 'liquid_soap']
		imgbw = []
		for index,value in enumerate(lb_list):
			ann_dic = {}
			imlab = cv2.imread(os.path.join(totalfilepath, label), cv2.IMREAD_GRAYSCALE)
			#cv2.imshow('label',imlab)
			#cv2.waitKey(0)
			#bg = imlab[:,:] == 0      #background
			lbrg = imlab[:,:] == value    #labelregion
			imlab[lbrg] = 255
			#cv2.imshow('mask',imlab)
			#cv2.waitKey(0)
			lbbg = imlab[:,:] != 255
			imlab[lbbg] = 0
			#cv2.imshow('mask',imlab)
			#cv2.waitKey(0)
			imlab = np.asarray(imlab[:,:],order='F')
			imgbw.append(imlab)
			rle = encode(imgbw[index])
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
				pose_dic = yaml.load(pose_para, Loader=yaml.FullLoader)
			pose_value = pose_dic[obj_list[index]]
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
			ann_dic["category_id"] = value
			ann_dic["id"] = j
			j += 1
			ann_gt.append(ann_dic)
		i += 1
	return img_gt, ann_gt

#function to dump json
def dump_json(image_set):
	result = {}
	result["info"] = {"description":None,"version":1.0,"year":2019,"contributor":"Huo","date_created":1558011617.454825}
	result["cameras"] = [{"name":"kinect1","K":[525,0.0,319.5,0.0,525,239.5,0.0,0.0,1.0],"height":480,"width":640,"camera_type":"kinect1","id":0}]
	result["licenses"] = [{"url":"https://opensource.org/licenses/BSD-3-Clause","id":0,"name":"BSD-3-Clause"}]
	result["images"] ,result["annotations"]= json_image_ann(image_set)
	result["categories"] = [{"supercategory":"objects","mesh":"kinect_0001.stl","id":1,"name":"kinect"},{"supercategory":"objects","mesh":"bottle_0001.stl","id":2,"name":"bottle"},{"supercategory":"objects","mesh":"cola_can_0001.stl","id":3,"name":"cola_can"},{"supercategory":"objects","mesh":"liquid_soap_0001.stl","id":4,"name":"liquid_soap"},]
	#print(result)
	return result

if __name__ == "__main__":

	shutil.copytree(sourcefilepath1, totalfilepath) 
	renameall(sourcefilepath2, totalfilepath)
	remove_color_label(totalfilepath)
	rgblist = get_file_list(totalfilepath, '_rgb', imagepath)
	depthlist = get_file_list(totalfilepath, 'epth', depthpath)
	#print(rgblist)
	#print(len(rgblist))
	images = os.listdir(imagepath)
	num = len(images)
	train_images = random.sample(images, int(train_percent*num))
	test_images = [item for item in images if item not in train_images]

	filename='20190517_Coke.json'
	result = dump_json(images)
	with open(filename,'w') as ong:
		json.dump(result,ong)

	filename='train_20190517_Coke.json'
	result = dump_json(train_images)
	with open(filename,'w') as ong:
		json.dump(result,ong)

	filename='val_20190517_Coke.json'
	result = dump_json(test_images)
	with open(filename,'w') as ong:
		json.dump(result,ong)