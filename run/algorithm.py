# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2023 Freescale Semiconductor
# Copyright 2024 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 5.0 
# * Code Date: 2024/12/26
# * Author   : Weilly Li
# * Inference and Draw , split using, For GTK GUI 
# * v4.0 : Add PostPrecess_YOLO_Ouput_tflite_V2 , PostPrecess_YOLO_Plot_Pose
# * v5.0 : Add detect_objectdetect_YOLOv8n_NXP & detect_palm_and_landmark
#--------------------------------------------------------------------------------------

import os
import csv
import cv2
import cairo
#import tflite_runtime.interpreter as tflite #[RK3588] Remove
from utils import *

facemask_dict =  {0: 'Mask', 1: 'NoMask'}
facemask_YOLOv5s_dict = {0:'Mask', 1:'No Mask', 2:'maybe weared incorrect'}
hardhat_YOLOv5s_dict  = {0:'head', 1:'helmet', 2:'hi-viz helmet', 3:'hi-viz vest', 4:'person'}
gender_dict = {0: "woman", 1: "man"}
ethnicity_dict = { 0:'asian', 1:'indian', 2:'black', 3:'white', 4:'middle eastern'}
emotion_dict = {0: "Neutral", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral_", 5: "Sad", 6: "Surprised"}
fruit_dict = {0: 'cucumber', 1: 'Apple', 2: 'kiwi', 3: 'banana', 4: 'orange', 5: 'coconut', 6 : 'peach', 7 :'cherry', 8 :'pear', 9:'pomegranate', 10:'pineapple', 11:'watermelon', 12:'melon', 13:'grape', 14:'strawberry'}
pcb_dict = {0: 'none', 1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'copper', 6: 'pin-hole'}
pcb_electron_dict = {0: 'IC', 1: 'LED', 2: 'battery', 3: 'buzzer', 4: 'capacitor', 5: 'clock', 6: 'condensator', 7: 'connector', 8: 'diode', 9: 'display', 10: 'fuse', 11: 'inductor', 12: 'led', 13: 'potentiometer', 14: 'relay', 15: 'resistor', 16: 'switch', 17: 'transistor'}
coco_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# ------------------------------------------------   API  ------------------------------------------------
def LoadingImage_OutputRGB565(img_path, w , h ):
	im = cv2.imread(img_path)
	im = cv2.resize(im, (w, h))
	R5 = (im[...,2]>>3).astype(np.uint16) << 11
	G6 = (im[...,1]>>2).astype(np.uint16) << 5
	B5 = (im[...,0]>>3).astype(np.uint16)
	RGB565 = R5 | G6 | B5
	return [RGB565, 1 , 1]

def LoadingImage_OutputRGB888(img_path, w , h ):
	im = cv2.imread(img_path)
	im = cv2.resize(im, (w, h))
	RGB888 = im
	return [RGB888, 1, 1] 

def GetDstBuffer(dst,dst_Width,dst_Height,IsLoadPicture):
	if IsLoadPicture==True :
		# Loading Image
		im = cv2.resize(dst, (dst_Width, dst_Height))
		R5 = (im[...,2]>>3).astype(np.uint16) << 11
		G6 = (im[...,1]>>2).astype(np.uint16) << 5
		B5 = (im[...,0]>>3).astype(np.uint16)
		RGB565 = R5 | G6 | B5
		return RGB565
	else :
		return dst

def PostPrecess_YOLO_Ouput_tflite_V1(interpreter,output_details):
	# Parameter
	width    = interpreter.get_input_details()[0]['shape'][2]
	height   = interpreter.get_input_details()[0]['shape'][1]

	# Modify Ouput [y]
	y = []
	for output in output_details:
		x = interpreter.get_tensor(output_details[0]['index'])
		if (interpreter.get_input_details()[0]['dtype']==np.uint8) : 
			scale, zero_point = output_details[0]['quantization']
			x = (x.astype(np.float32) - zero_point) * scale  # re-scale
		y.append(x)
	y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
	y[0][..., :4] *= [width, height, width, height]
	return y

def PostPrecess_YOLO_Ouput_tflite_V2(interpreter,output_details, IsPoseTask):
	# Parameter
	width    = interpreter.get_input_details()[0]['shape'][2]
	height   = interpreter.get_input_details()[0]['shape'][1]
	
	# Modify y (output) after Tensorflow Lite
	y = []
	for output in output_details: # Yolov8 modify , [WPI]
		x = interpreter.get_tensor(output_details[0]['index'])
		if x.ndim == 3:
			if x.shape[-1] == 6:
				x[:, :, [0, 2]] *= width
				x[:, :, [1, 3]] *= height
			else:
				x[:, [0, 2]] *= width
				x[:, [1, 3]] *= height
				if IsPoseTask: #Pose , # Yolov8 modify , [WPI]
					x[:, 5::3] *= width
					x[:, 6::3] *= height
		y.append(x)

	if len(y) == 2:
		if len(y[1].shape) != 4:
			y = list(reversed(y))
			y[1] = np.transpose(y[1], (0, 3, 1, 2))

	y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

	return y

def PostPrecess_YOLO_Plot_Box(context, preds, annotator, Labels, DstShape ,BaseShape):
	for idx, pred in enumerate(preds):
		if len(pred):
			pred[:, :4] = scale_boxes([DstShape[1], DstShape[0]], pred[:, :4], BaseShape).round()
			conf = torch.flip(pred[:, 4], dims=[0])
	
			i = 0 
			for *xyxy, conf_, cls in reversed(pred):
				c = int(cls)  # integer class
				label = f"{Labels[c]} , {conf[i]:.2f}"
				annotator.box_label(context, xyxy, label, color=colors_(c, True))
				i+=1

def PostPrecess_YOLO_Plot_Pose(context, preds, annotator, DstShape ,BaseShape):
	# Plot Pose results
	kpt_shape=[17, 3]
	for i, pred in enumerate(preds):
		if len(pred):
			pred_kpts = pred[:, 6:].view(len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
			pred_kpts = scale_coords([DstShape[1], DstShape[0]], pred_kpts, BaseShape)
			# Plot Pose results
			for k in reversed(pred_kpts):
				annotator.kpts(context, k, BaseShape, radius=3, kpt_line=True)



# ---------------------------------------------   Algorithm  ---------------------------------------------


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------   (1) Object   --------------------------------------------
# --------------------------------------------------------------------------------------------------------
# DEMO 1.1 : Objection detection
# * Code Ver : 3.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_objectdetect_mobilnetv2_ssd(interpreterModel, frame, PicturePath, IsLoadPicture):
    
	# Capture Image
	if IsLoadPicture==True : 
		FrameInput = cv2.imread(PicturePath)
	else :
		FrameInput = frame

	# define
	obj_threshold       = 0.5
	labels_coco         = load_labels("../data/label/coco_labels.txt") 
	colors_coco         = generate_colors(labels_coco)

	# Inference
	output_Info    = RunInferenceMode( interpreterModel, FrameInput ) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 
	positions      = np.squeeze(interpreterModel.get_tensor(output_details[0]['index']))
	classes        = np.squeeze(interpreterModel.get_tensor(output_details[1]['index']))
	scores         = np.squeeze(interpreterModel.get_tensor(output_details[2]['index']))

	# object 
	result = []
	for idx, score in enumerate(scores):
		if score > obj_threshold:
			result.append({'pos': positions[idx], '_id': classes[idx]})
	
	# object output
	object_details = []
	for obj in result:
		pos = obj['pos']
		_id = obj['_id']
		x1 = int(pos[1] * frame.shape[1])
		x2 = int(pos[3] * frame.shape[1])
		y1 = int(pos[0] * frame.shape[0])
		y2 = int(pos[2] * frame.shape[0])
		top    = max(0, np.floor(y1 + 0.5).astype('int32'))
		left   = max(0, np.floor(x1 + 0.5).astype('int32'))
		bottom = min(frame.shape[0], np.floor(y2 + 0.5).astype('int32'))
		right  = min(frame.shape[1], np.floor(x2 + 0.5).astype('int32'))
		color = colors_coco[int(_id) % len(colors_coco)]; 
		object_details.append([labels_coco[_id], color, left, top, right, bottom])

	# OpenCV(BGR888 to RGB565) into Cairo Surface
	if IsLoadPicture==True :
		# Loading Image
		im = cv2.resize(FrameInput, (frame.shape[1], frame.shape[0]))
		R5 = (im[...,2]>>3).astype(np.uint16) << 11
		G6 = (im[...,1]>>2).astype(np.uint16) << 5
		B5 = (im[...,0]>>3).astype(np.uint16)
		RGB565 = R5 | G6 | B5
		FrameOutput = RGB565
	else :
		# Camera Streamer
		FrameOutput = frame

	return [FrameOutput, object_details, InfereneTime]
def draw_objectdetect_mobilnetv2_ssd(context, frame, drawout_details):
	
	# define
	rect_linewidth      = 4
	label_text_fontsize = 24

	# draw
	for obj in drawout_details :
		label = obj[0]
		color = obj[1]
		left  = obj[2]
		top   = obj[3]
		right = obj[4]
		bottom= obj[5]		
		x = left; y = top; 
		w = (right-left); h=(bottom-top); 
		line_width=rect_linewidth; font_size = label_text_fontsize
		DrawObjRect_cairo(context, x, y, w, h, color, line_width)
		DrawObjLabelText_cairo(context, x, y, color, label, font_size)
def draw_objectdetect_mobilnetv2_ssd_ByDaulCamera(context, frame, drawout_details):
	
	# define
	rect_linewidth      = 4
	label_text_fontsize = 24

	# draw
	for obj in drawout_details :
		label = obj[0]
		color = obj[1]
		left  = obj[2]/2
		top   = obj[3]
		right = obj[4]/2
		bottom= obj[5]		
		x = left; y = top; 
		w = (right-left); h=(bottom-top); 
		line_width=rect_linewidth; font_size = label_text_fontsize
		DrawObjRect_cairo(context, x, y, w, h, color, line_width)
		DrawObjLabelText_cairo(context, x, y, color, label, font_size)	

# --------------------------------------------------------------------------------------------------------
# DEMO 1.2 : Objection detection (YOLOv5s)
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_objectdetect_YOLOv5s(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Param Setting
	obj_threshold = 0.2
	IoU           = 0.45

	# Capture Image
	if IsLoadPicture==True :
		FrameInput = cv2.imread(PicturePath)
	else :
		FrameInput = frame

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	output_Info    = RunInferenceMode_YOLO( interpreterModel, FrameInput ) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# Inference of output
	y = PostPrecess_YOLO_Ouput_tflite_V1(interpreterModel,output_details)
	
	# Object Filter by Using NMS
	pred = non_max_suppression_YOLOv5(torch.from_numpy(y[0]), obj_threshold, IoU, None, False, max_det=1000)

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , pred], InfereneTime]
def draw_objectdetect_YOLOv5s(context, frame, drawout_details):
	# DrawOut
	preds  = drawout_details[1]
	annotator = Annotator(frame, line_width=3)
	PostPrecess_YOLO_Plot_Box(context, preds, annotator, coco_dict, [drawout_details[0][0],drawout_details[0][1]], frame.shape)

# --------------------------------------------------------------------------------------------------------
# DEMO 1.3 : Objection detection (YOLOv8n)
# * Code Ver : 1.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_objectdetect_YOLOv8n_NXP(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Param Setting
	obj_threshold = 0.2
	IoU           = 0.45

	# Capture Image
	if IsLoadPicture==True :
		FrameInput = cv2.imread(PicturePath)
	else :
		FrameInput = frame

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	output_Info    = RunInferenceMode_YOLO( interpreterModel, FrameInput ) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 
	out_scale, out_zero_point = output_details[0]["quantization"]

	# Get y of output
	y = interpreterModel.get_tensor(output_details[0]['index'])
	y = (y.astype(np.float32) - out_zero_point) * out_scale
	y[:, 0] -= 0.0 #pad
	y[:, 1] -= 0.2187
	y[:, :4] *= max(frame.shape) ; y = y.transpose(0, 2, 1)
	y[..., 0] -= y[..., 2] / 2
	y[..., 1] -= y[..., 3] / 2

	# Object Filter by Using NMS
	pred = []
	for obj in y:
		scores = obj[:, 4:].max(-1)
		keep = scores > obj_threshold
		boxes = obj[keep, :4]
		scores = scores[keep]
		if len(keep) > 0:
			class_ids = obj[keep, 4:].argmax(-1)
			indices = cv2.dnn.NMSBoxes(boxes, scores, obj_threshold,  IoU)
			if indices is not None and len(indices) > 0:
				indices = indices.flatten()
				for i in indices:
					pred.append({"box": boxes[i].tolist(),  "score": float(scores[i]),  "class_id": int(class_ids[i]) })
	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , pred], InfereneTime]
def draw_objectdetect_YOLOv8n_NXP(context, frame, drawout_details):
	# DrawOut
	pred  = drawout_details[1]
	labels_coco   = load_labels("../data/label/coco_labels.txt") 
	colors_coco   = generate_colors(labels_coco)
	rect_linewidth      = 4
	label_text_fontsize = 24
	for detection in pred:
		box = detection["box"]  
		score = detection["score"]  
		class_id = detection["class_id"]
		x, y, w, h = box
		color = colors_coco[int(class_id) % len(colors_coco)]; 
		label = coco_dict[class_id]
		line_width=rect_linewidth; font_size = label_text_fontsize
		DrawObjRect_cairo(context, x, y, w, h, color, line_width)
		DrawObjLabelText_cairo(context, x, y, color, label, font_size)

# --------------------------------------------------------------------------------------------------------
# DEMO 1.3 : Fruit detection (YOLOv5s) 
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_fruitdetect_YOLOv5s(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Param Setting
	obj_threshold = 0.25
	IoU           = 0.45
	
	# Capture Image
	if IsLoadPicture==True :
		FrameInput  = cv2.imread(PicturePath)
	else:
		FrameInput  = frame

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	input_details  = interpreterModel.get_input_details()
	output_Info    = RunInferenceMode_YOLO(interpreterModel, FrameInput) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# Inference of output
	y = PostPrecess_YOLO_Ouput_tflite_V1(interpreterModel,output_details)
	
	# Obejct filter by uisng NMS
	pred = non_max_suppression_YOLOv5(torch.from_numpy(y[0]), obj_threshold, IoU, None, False, max_det=1000)
	
	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , pred], InfereneTime]
def draw_fruitdetect_YOLOv5s(context, frame, drawout_details):
	preds  = drawout_details[1]
	annotator = Annotator(frame, line_width=3)
	PostPrecess_YOLO_Plot_Box(context, preds, annotator, fruit_dict, [drawout_details[0][0],drawout_details[0][1]], frame.shape)

# --------------------------------------------------------------------------------------------------------
# DEMO 1.4 : Mask Detector (YOLOv5s)
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_facemask_YOLOv5s(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Param Setting
	obj_threhsold         = 0.5
	IoU                   = 0.6

	# Capture Image
	if IsLoadPicture==True :
		FrameInput  = cv2.imread(PicturePath)
	else:
		FrameInput  = frame

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	numpy_type     = interpreterModel.get_input_details()[0]['dtype']
	output_Info    = RunInferenceMode_YOLO(interpreterModel, FrameInput) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# Inference of output
	y = PostPrecess_YOLO_Ouput_tflite_V1(interpreterModel,output_details)
	
	# Outputs Filter by NMS
	pred = non_max_suppression_YOLOv5(torch.from_numpy(y[0]), obj_threhsold, IoU, None, False, max_det=1000)
	
	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , pred], InfereneTime]
def draw_facemask_YOLOv5s(context, frame, drawout_details):
	#define
	mask_rect_color      = (0,255,0)
	unmask_rect_color    = (0,0,255)
	warning_rect_color   = (0,180,255)
	
	# Build Output Result
	pred   = drawout_details[1]
	width  = drawout_details[0][0]
	height = drawout_details[0][1]
	for idx, det in enumerate(pred):
		annotator = Annotator(frame, line_width=3)
		if len(det):
			det[:, :4] = scale_boxes([width, height], det[:, :4], frame.shape).round()
		
		# Add bbox to image
		for *xyxy, conf, cls in reversed(det):
			c = int(cls)  # integer class
			label = facemask_YOLOv5s_dict[c]
			if label=='Mask':
				annotator.box_label(context, xyxy, label, color=mask_rect_color)
			if label=='No Mask':
				annotator.box_label(context, xyxy, label, color=unmask_rect_color)
			if label=='maybe weared incorrect':
				annotator.box_label(context, xyxy, label, color=warning_rect_color)

# --------------------------------------------------------------------------------------------------------
# DEMO 1.5 : PCB detection (YOLOv5s)
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_PCBdetect_YOLOv5s(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Param Setting
	Conf =0.25
	IoU = 0.45
	detMax = 1000

	# Capture Image
	if IsLoadPicture==True :
		FrameInput   = cv2.cvtColor(cv2.imread(PicturePath), cv2.COLOR_BGR2RGB)
	else:
		FrameInput   = frame

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	input_details  = interpreterModel.get_input_details()
	output_Info    = RunInferenceMode_YOLO(interpreterModel, FrameInput) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# Inference of output
	y = PostPrecess_YOLO_Ouput_tflite_V1(interpreterModel,output_details)
	
	# Object Fileter by uinsg NMS
	pred = non_max_suppression_YOLOv5(torch.from_numpy(y[0]), Conf, IoU, None, False, max_det=detMax)
	
	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , pred], InfereneTime]
def draw_PCBdetect_YOLOv5s(context, frame, drawout_details):
	pred   = drawout_details[1]
	width  = drawout_details[0][0]
	height = drawout_details[0][1]
	for idx, det in enumerate(pred):
		annotator = Annotator(frame, line_width=3)
		if len(det):
			det[:, :4] = scale_boxes([width, height], det[:, :4], frame.shape).round()

		# Add bbox to image
		for *xyxy, conf, cls in reversed(det):
			c = int(cls)  # integer class
			label = pcb_dict[c]
			annotator.box_label(context, xyxy, label, color=colors_(c, True))

# --------------------------------------------------------------------------------------------------------
# DEMO 1.7 : PCB detection (YOLOv8s)
# * Code Ver : 1.0
# * Code Date: 2024/12/20
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_PCBdetect_YOLOv8s(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Param Setting
	obj_threshold  =0.25
	IoU = 0.45
	detMax = 100

	# Capture Image
	if IsLoadPicture==True :
		FrameInput = cv2.imread(PicturePath)
		FrameInputRGB = FrameInput 
	else:
		FrameInput   = frame
		FrameInputRGB= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	input_details  = interpreterModel.get_input_details()
	output_Info    = RunInferenceMode_YOLO(interpreterModel, FrameInputRGB) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# Inference of Ouput
	y = PostPrecess_YOLO_Ouput_tflite_V2(interpreterModel, output_details, False)

	# NMS 
	preds = non_max_suppression_YOLOv8(prediction=torch.from_numpy(y[0]), conf_thres=0.25, iou_thres=0.45, max_det=3000, max_time_img=0.1)

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , preds], InfereneTime]
def draw_PCBdetect_YOLOv8s(context, frame, drawout_details):
	pred   = drawout_details[1]
	width  = drawout_details[0][0]
	height = drawout_details[0][1]
	for idx, det in enumerate(pred):
		annotator = Annotator(frame, line_width=5)
		if len(det):
			det[:, :4] = scale_boxes([width, height], det[:, :4], frame.shape).round()

		# Add bbox to image
		for *xyxy, conf, cls in reversed(det):
			c = int(cls)  # integer class
			label = pcb_electron_dict[c]
			annotator.box_label(context, xyxy, label, color=colors_(c, True))


# --------------------------------------------------------------------------------------------------------
# DEMO 1.8 : ADAS detection  (TBC , lane borke)
# * Code Ver : 1.0
# * Code Date: 2023/03/08
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
cfg = ModelConfig()
mean_lane=[0.485, 0.456, 0.406]
std_lane =[0.229, 0.224, 0.225]
def process_laneline(output, cfg):
	# Parse the output of the model to get the lane informatio
	processed_output = output[:, ::-1, :]
	prob = softmax(processed_output[:-1, :, :])#scipy.special.softmax(processed_output[:-1, :, :], axis=0)
	idx = np.arange(cfg.griding_num) + 1
	idx = idx.reshape(-1, 1, 1)
	loc = np.sum(prob * idx, axis=0)
	
	processed_output = np.argmax(processed_output, axis=0)
	loc[processed_output == cfg.griding_num] = 0
	processed_output = loc
	
	col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
	col_sample_w = col_sample[1] - col_sample[0]
	lane_points_mat = []
	lanes_detected = []
	max_lanes = processed_output.shape[1]
	for lane_num in range(max_lanes):
		lane_points = []
		# Check if there are any points detected in the lane
		if np.sum(processed_output[:, lane_num] != 0) > 2:
			lanes_detected.append(True)
			# Process each of the points for each lane
			for point_num in range(processed_output.shape[0]):
				if processed_output[point_num, lane_num] > 0:
					lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
					lane_points.append(lane_point)
		else:
			lanes_detected.append(False)
		
	lane_points_mat.append(lane_points)

	return np.array(lane_points_mat), np.array(lanes_detected)
def detect_ADAS(interpreterModel, interpreterModel_Sub ,frame, PicturePath, IsLoadPicture):
	# SWITCH 
	LDW = False
	Vehicle = True


	# Capture Image
	if IsLoadPicture==True :
		# Loading Image
		FrameInput = cv2.imread(PicturePath)
		FrameInputRGB  = cv2.cvtColor(FrameInput, cv2.COLOR_BGR2RGB)
	else:
		# Camera Streamer
		FrameInput =  frame
		FrameInputRGB  = frame

	# --------------------------------------------------------------------------------------------------------
	# Lane Detection
	# --------------------------------------------------------------------------------------------------------
	lanes_points = []
	lanes_detected = []

	if LDW :
		input_details  = interpreterModel.get_input_details()
		output_details = interpreterModel.get_output_details()
		width          = input_details[0]['shape'][2]
		height         = input_details[0]['shape'][1]
		nChannel       = input_details[0]['shape'][3]
	
		frame_resized  = cv2.resize(FrameInputRGB, (width, height))
		input_data     = ((frame_resized/ 255.0 - mean_lane) / std_lane).astype(np.float32)
		input_data     = input_data[np.newaxis,:,:,:] 
		interpreterModel.set_tensor(input_details[0]['index'], input_data) 
		interpreter_time_start = time.time()
		interpreterModel.invoke()
		interpreter_time_end   = time.time()
		InfereneTime_LDW  = interpreter_time_end - interpreter_time_start
		num_anchors 	  = interpreterModel.get_output_details()[0]['shape'][1]
		num_lanes   	  = interpreterModel.get_output_details()[0]['shape'][2]
		num_points  	  = interpreterModel.get_output_details()[0]['shape'][3]
		LaneLine          = interpreterModel.get_tensor(output_details[0]['index']).reshape(num_anchors, num_lanes, num_points)
		lanes_points, lanes_detected = process_laneline(LaneLine, cfg)
	else :
		InfereneTime_LDW = 0

	
	# --------------------------------------------------------------------------------------------------------
	# Vehicle Detection
	# --------------------------------------------------------------------------------------------------------
	Obejct_details = []
	if Vehicle :
		# define
		labels_coco            = load_labels("../data/label/coco_labels.txt") 
		colors_coco            = generate_colors(labels_coco)
		interpreter_time_start = time.time()
		output_vehicle         = RunInferenceMode(interpreterModel_Sub, FrameInputRGB) 
		interpreter_time_end   = time.time()
		InfereneTime_Vehicle   = interpreter_time_end - interpreter_time_start
		output_details_vehicle = output_vehicle[0] ; 
		positions              = np.squeeze(interpreterModel_Sub.get_tensor(output_details_vehicle[0]['index']))
		classes                = np.squeeze(interpreterModel_Sub.get_tensor(output_details_vehicle[1]['index']))
		scores                 = np.squeeze(interpreterModel_Sub.get_tensor(output_details_vehicle[2]['index']))

		# object 
		result = []
		for idx, score in enumerate(scores):
			if score > 0.5 :
				result.append({'pos': positions[idx], '_id': classes[idx]})
		
		# object output
		for obj in result:
			pos = obj['pos']
			_id = obj['_id']
			if labels_coco[_id]=="car" or labels_coco[_id]=="truck" or labels_coco[_id]=="train" or \
				labels_coco[_id]=="bus" or labels_coco[_id]=="motorcycle" or labels_coco[_id]=="person" :
				x1 = int(pos[1] * frame.shape[1])
				x2 = int(pos[3] * frame.shape[1])
				y1 = int(pos[0] * frame.shape[0])
				y2 = int(pos[2] * frame.shape[0])
				top    = max(0, np.floor(y1 + 0.5).astype('int32'))
				left   = max(0, np.floor(x1 + 0.5).astype('int32'))
				bottom = min(frame.shape[0], np.floor(y2 + 0.5).astype('int32'))
				right  = min(frame.shape[1], np.floor(x2 + 0.5).astype('int32'))
				color = colors_coco[int(_id) % len(colors_coco)]; 
				Obejct_details.append([labels_coco[_id], color, left, top, right, bottom])
	else :
		InfereneTime_Vehicle = 0

	InfereneTime = InfereneTime_LDW + InfereneTime_Vehicle

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [FrameOutput, [lanes_points, lanes_detected, Obejct_details], InfereneTime]
def draw_ADAS(context, frame, drawout_details):
	# Define
	lane_points_mat  = drawout_details[0]
	lanes_detected   = drawout_details[1]
	Obejct_details   = drawout_details[2]
	LDW = False
	Vehicle = True
	context.set_line_width(2)
	context.set_source_rgb(1,0,0)

	if LDW :
		# draw lane
		circle_linewidth = 3
		for lane_num,lane_points in enumerate(lane_points_mat):
			for lane_point in lane_points:
				context.arc( lane_point[0], lane_point[1], circle_linewidth, 0, 2 * 3.14159)
				context.fill()	

	if Vehicle :
		# draw Vehicle & Obejct
		rect_linewidth      = 4
		label_text_fontsize = 24
		for obj in Obejct_details :
			label = obj[0]
			color = obj[1]
			left  = obj[2]
			top   = obj[3]
			right = obj[4]
			bottom= obj[5]		
			x = left; y = top; 
			w = (right-left); h=(bottom-top); 
			line_width=rect_linewidth; font_size = label_text_fontsize
			DrawObjRect_cairo(context, x, y, w, h, color, line_width)
			DrawObjLabelText_cairo(context, x, y, color, label, font_size)

# --------------------------------------------------------------------------------------------------------
# DEMO 1.7 : HardHat Detector (YOLOv5s)
# * Code Ver : 1.0
# * Code Date: 2023/09/11
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_hardhat_YOLOv5s(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Param Setting
	Conf =0.4
	IoU = 0.3
	detMax = 300

	# Capture Image
	if IsLoadPicture==True :
		# Loading Image
		FrameInput = cv2.imread(PicturePath)
	else:
		# Camera Streamer
		FrameInput = frame

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	numpy_type     = interpreterModel.get_input_details()[0]['dtype']
	output_Info    = RunInferenceMode_YOLO(interpreterModel, FrameInput) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# Inference of output
	y = PostPrecess_YOLO_Ouput_tflite_V1(interpreterModel,output_details)
	
	# Outputs Filter by NMS
	pred = non_max_suppression_YOLOv5(torch.from_numpy(y[0]), Conf, IoU, None, False, max_det=detMax)
	
	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , pred], InfereneTime]
def draw_hardhat_YOLOv5s(context, frame, drawout_details):
	#define
	helmet_rect_color  = (0,255,0)
	head_rect_color    = (0,0,255)
	hi_viz_helmet_rect_color  = (0,180,255)
	hi_viz_vest_rect_color    = (50,180,255)
	person_color              = (150,150,150)
	
	# Build Output Result
	pred   = drawout_details[1]
	width  = drawout_details[0][0]
	height = drawout_details[0][1]
	for idx, det in enumerate(pred):
		annotator = Annotator(frame, line_width=3)
		if len(det):
			det[:, :4] = scale_boxes([width, height], det[:, :4], frame.shape).round()
		
		# Add bbox to image
		for *xyxy, conf, cls in reversed(det):
			c = int(cls)  # integer class
			label = hardhat_YOLOv5s_dict[c]
			if label=='helmet':
				annotator.box_label(context, xyxy, label, color=helmet_rect_color)
			if label=='head':
				annotator.box_label(context, xyxy, label, color=head_rect_color)
			if label=='hi-viz helmet':
				annotator.box_label(context, xyxy, label, color=hi_viz_helmet_rect_color)
			if label=='hi-viz vest':
				annotator.box_label(context, xyxy, label, color=hi_viz_vest_rect_color)
			if label=='person':
				annotator.box_label(context, xyxy, label, color=person_color)

# --------------------------------------------------------------------------------------------------------
# DEMO TBC : face detection
# * Code Ver : 3.0
# * Code Date: 2023/09/08
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_facedetect_mobilnetv2_ssd(interpreterModel, frame, PicturePath, IsLoadPicture):

	# Capture Image
	if IsLoadPicture==True :
		FrameInput = cv2.imread(PicturePath)
	else :
		FrameInput = frame
	
	# Info
	IoU                = 0.5
	adj_x_scale        = 0.1
	adj_y_scale        = 0.05
	offset_y           = 5
	obj_threhsold      = 0.6

	# --------------------------------------------------------------------------------------------------------
	# Face Detection
	# --------------------------------------------------------------------------------------------------------
	# Inference 
	output_Info       = RunInferenceMode(interpreterModel, FrameInput) 
	output_details    = output_Info[0] ; InfereneTime = output_Info[1] 
	detection_boxes   = interpreterModel.get_tensor(output_details[0]['index'])
	detection_classes = interpreterModel.get_tensor(output_details[1]['index'])
	detection_scores  = interpreterModel.get_tensor(output_details[2]['index'])
	num_boxes         = interpreterModel.get_tensor(output_details[3]['index'])
	boxs              = np.squeeze(detection_boxes)
	scores            = np.squeeze(detection_scores) #(10,)
	boxs_nms, scores_nms = nms(boxs, scores, float(IoU))

	# Outputs
	output_details_Face = []
	for i in range( 0, len(scores_nms)-1) :
		if scores_nms[i] > obj_threhsold: 
			#  Face
			x = boxs_nms[i, [1, 3]] * frame.shape[1]
			y = boxs_nms[i, [0, 2]] * frame.shape[0]
			x[1] = x[1] + int((x[1]-x[0])*adj_x_scale)
			y[1] = y[1] + int((y[1]-y[0])*adj_y_scale) 
			output_details_Face.append([x,y])

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [FrameOutput, output_details_Face, InfereneTime]
def draw_facedetect_mobilnetv2_ssd(context, frame, drawout_details):
	# define
	face_rect_color    = (0, 255, 0)
	face_label_color   = (200, 160, 80)
	face_rect_width    = 4
	face_circle_radius = 2
	label_text_fontsize= 24

	# Draw Face
	faces = drawout_details
	for face in faces :
		x = face[0]
		y = face[1]
		w = int(x[1]-x[0])
		h = int(y[1]-y[0])
		DrawObjRect_cairo(context, int(x[0]),  int(y[0]), w, h, face_rect_color, face_rect_width)
		DrawObjLabelText_cairo(context, int(x[0]),  int(y[0]) , face_rect_color, "face", label_text_fontsize)

# --------------------------------------------------------------------------------------------------------
# DEMO TBC : Fruit detection_MobileNet , x
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_fruitdetect_mobilenetv2_ssd(interpreterModel, frame):

	# Define
	IoU           = 0.4
	labels_friut  = load_labels("../data/label/friut_labels.txt") 
	colors_friut  = generate_colors(labels_friut)

	# get interpreter result
	frame_rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_input_data = ((frame_rgb.astype("float32"))-128)/128
	output_details   = RunInferenceMode(interpreterModel, frame_input_data)
	boxs             = np.squeeze(interpreterModel.get_tensor(output_details[0]['index']))
	scores           = np.squeeze(interpreterModel.get_tensor(output_details[1]['index']))

	# output - list
	drawout_details = []
	for class_id in range(1,len(labels)-1):
		boxs_nms, scores_nms = nms(boxs, scores.swapaxes(0,1)[class_id], float(IoU))
		idx = 0
		for score in scores_nms:
			if (np.max(score) > 0.5) :
				x0 = int(boxs_nms[idx][0]*frame.shape[1])
				y0 = int(boxs_nms[idx][1]*frame.shape[0])
				x1 = int(boxs_nms[idx][2]*frame.shape[1])
				y1 = int(boxs_nms[idx][3]*frame.shape[0])
				x = x0; y = y0; w = (x1-x0); h=(y1-y0); color = colors_friut[int(class_id) % len(colors_friut)]; line_width=4; font_size = 2;
				drawout_details.append([labels[class_id], color, x0, x1, y0, y1])			
		idx = idx + 1
		
	return [ frame, drawout_details, InfereneTime]
def draw_fruitdetect_mobilenetv2_ssd(context, frame, drawout_details):
	
	# define
	rect_linewidth      = 4
	label_text_fontsize = 24

	# draw
	for obj in drawout_details :
		label = obj[0]
		color = obj[1]
		left  = obj[2]
		top   = obj[3]
		right = obj[4]
		bottom= obj[5]		
		x = left; y = top; 
		w = (right-left); h=(bottom-top); 
		line_width=rect_linewidth; font_size = label_text_fontsize
		DrawObjRect_cairo(context, x, y, w, h, color, line_width)
		DrawObjLabelText_cairo(context, x, y, color, label, font_size)



# --------------------------------------------------------------------------------------------------------
# ------------------------------------------   (2) Segmentation   ----------------------------------------
# --------------------------------------------------------------------------------------------------------
# DEMO 2.1 : Objection Segmentation (YOLOv5s)
# * Code Ver : 1.0
# * Code Date: 2023/07/17
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_segmentation_YOLOv5s(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Capture Image
	if IsLoadPicture==True :
		FrameInput = cv2.imread(PicturePath)
	else :
		FrameInput = frame

	# Inference
	width             = interpreterModel.get_input_details()[0]['shape'][2]
	height            = interpreterModel.get_input_details()[0]['shape'][1]
	input_details     = interpreterModel.get_input_details()
	output_details    = interpreterModel.get_output_details()#output_details    = RunInferenceMode_YOLO(interpreterModel, frame)
	scale, zero_point = input_details[0]['quantization']

	frame_resized     = letterbox(FrameInput, (width,height), stride=32, auto=False)[0]  # padded resize
	frame_resized     = frame_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
	frame_resized     = np.ascontiguousarray(frame_resized)

	im = torch.from_numpy(frame_resized).to('cpu').float()
	im /= 255 # 0 - 255 to 0.0 - 1.0
	if len(im.shape) == 3:
		im = im[None]  # expand for batch dim

	if (input_details[0]['dtype']==np.uint8) :
		input_data = (im.permute(0, 2, 3, 1).cpu().numpy() / scale + zero_point).astype(np.uint8)
	else :
		input_data = im.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)  
		
	interpreterModel.set_tensor(input_details[0]['index'], input_data ) 
	interpreter_time_start = time.time()
	interpreterModel.invoke()
	interpreter_time_end   = time.time()
	InfereneTime = interpreter_time_end - interpreter_time_start

    # Object
	y = []
	for output in output_details:
		x = interpreterModel.get_tensor(output['index'])
		if (input_details[0]['dtype']==np.uint8) : 
			scale, zero_point = output['quantization']
			x = (x.astype(np.float32) - zero_point) * scale  # re-scale
		y.append(x)
	y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
	y[0][..., :4] *= [width, height, width, height]
	
	# Object Filter by Using NMS
	obj_threshold = 0.25
	IoU           = 0.45
	pred = non_max_suppression_YOLOv5(torch.from_numpy(y[0]), obj_threshold, IoU, None, False, max_det=1000, nm=32)
	proto = torch.from_numpy(y[1])
	
	# Object Segmetation
	seen = 0
	retina_masks = False
	for i, det in enumerate(pred):
		seen += 1
		annotator = Annotator(frame, line_width=3)
		if len(det):
			masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
			det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

			# Mask plotting
			annotator.masks(masks,
				colors=[colors_(x, True) for x in det[:, 5]],
				im_gpu=torch.as_tensor(frame, dtype=torch.float16).to('cpu').permute(2, 0, 1).flip(0).contiguous() /
				255 if retina_masks else im[i])
	

	# OpenCV into Cairo Surface
	im = annotator.result() 
	RGB888 = im
	FrameOutput = RGB888

	return [ FrameOutput, [[width,height] , pred, proto], InfereneTime]
def draw_segmentation_YOLOv5s(context, frame, drawout_details):
	# define
	pred   = drawout_details[1]
	proto  = drawout_details[2]
	width  = drawout_details[0][0]
	height = drawout_details[0][1]

	# Build Output Result
	seen = 0
	retina_masks = False
	for idx, det in enumerate(pred):
		annotator = Annotator(frame, line_width=3)
		if len(det):
			# Add bbox to image
			for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
				c = int(cls)  # integer class
				label = coco_dict[c]
				annotator.box_label(context, xyxy, label, color=colors_(c, True))


# --------------------------------------------------------------------------------------------------------
# -----------------------------------------   (3) Pose / Feature   ---------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# DEMO 3.1 : Face Mesh
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_facemesh_mediapipe(interpreterModel, interpreterModel_Sub, frame, PicturePath, IsLoadPicture):
	
	# Capture Image
	if IsLoadPicture==True :
		FrameInput   = cv2.imread(PicturePath)
	else:
		FrameInput  = frame

	# Info
	IoU                = 0.5
	adj_x_scale        = 0.1
	adj_y_scale        = 0.05
	offset_y           = 5
	obj_threhsold      = 0.6
	mesh_point         = 468

	# --------------------------------------------------------------------------------------------------------
	# Face Detection
	# --------------------------------------------------------------------------------------------------------
	# Inference 
	output_Info       = RunInferenceMode(interpreterModel, FrameInput) 
	output_details    = output_Info[0] ; InfereneTime = output_Info[1] 
	detection_boxes   = interpreterModel.get_tensor(output_details[0]['index'])
	detection_classes = interpreterModel.get_tensor(output_details[1]['index'])
	detection_scores  = interpreterModel.get_tensor(output_details[2]['index'])
	num_boxes         = interpreterModel.get_tensor(output_details[3]['index'])
	boxs              = np.squeeze(detection_boxes)
	scores            = np.squeeze(detection_scores) #(10,)
	boxs_nms, scores_nms = nms(boxs, scores, float(IoU))

	# Outputs
	output_details_Face = []
	output_details_FaceMesh  = []
	for i in range( 0, len(scores_nms)-1) :
		if scores_nms[i] > obj_threhsold: 

			#  Face
			x = boxs_nms[i, [1, 3]] * frame.shape[1]
			y = boxs_nms[i, [0, 2]] * frame.shape[0]
			x[0] = x[0] - int((x[1]-x[0])*adj_x_scale)
			x[1] = x[1] + int((x[1]-x[0])*adj_x_scale)
			y[0] = y[0] - int((y[1]-y[0])*adj_x_scale) 
			y[1] = y[1] + int((y[1]-y[0])*adj_y_scale) 
			output_details_Face.append([x,y])

			# --------------------------------------------------------------------------------------------------------
			# FaceMesh Detection
			# --------------------------------------------------------------------------------------------------------
			if (len(output_details_Face)==1) : # Just Procesing Frist
				# Face Image
				roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
				roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
				roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
				roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))
				face_img                = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1]
				face_input_data         = ((face_img.astype("float32"))-128)/128 # [-0.5,0.5] -> [-1, 1]

				# Inference
				FaceMesh_input_width    = interpreterModel_Sub.get_input_details()[0]['shape'][2]
				FaceMesh_input_height   = interpreterModel_Sub.get_input_details()[0]['shape'][1]
				FaceMesh_output         = RunInferenceMode(interpreterModel_Sub, face_input_data)
				FaceMesh_output_details = FaceMesh_output[0]; InfereneTime = InfereneTime + FaceMesh_output[1] 
				mesh                    = interpreterModel_Sub.get_tensor(FaceMesh_output_details[0]['index']).reshape(mesh_point, 3)
				size_rate               = [face_img.shape[1]/FaceMesh_input_width, face_img.shape[0]/FaceMesh_input_height]

				# FaceMesh coordinate
				for pt in mesh:
					x_mesh = int(roi_x0 + pt[0]*size_rate[0])
					y_mesh = int(roi_y0 + pt[1]*size_rate[1])
					output_details_FaceMesh.append([x_mesh,y_mesh])

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [FrameOutput,[output_details_Face, output_details_FaceMesh], InfereneTime]
def draw_facemesh_mediapipe(context, frame, drawout_details):

	# define
	mesh_color_white   = (200, 200, 200)
	face_rect_color    = (0, 255, 0)
	face_mesh_color    = (255, 255, 255)
	face_rect_width    = 4
	face_circle_radius = 2
	label_text_fontsize= 24

	# Draw Face
	faces = drawout_details[0]
	facemeshs = drawout_details[1]
	for face in faces :
		x = face[0]
		y = face[1]
		w = int(x[1]-x[0])
		h = int(y[1]-y[0])
		DrawObjRect_cairo(context, int(x[0]),  int(y[0]), w, h, face_rect_color, face_rect_width)
		DrawObjLabelText_cairo(context, int(x[0]),  int(y[0]) , face_rect_color, "face", label_text_fontsize)

	# Draw FaceMesh
	for facemesh in facemeshs :
		context.set_source_rgb(face_mesh_color[0]/255,face_mesh_color[1]/255,face_mesh_color[2]/255)
		x_mesh = facemesh[0]
		y_mesh = facemesh[1]
		context.arc( x_mesh, y_mesh, face_circle_radius, 0, 2 * 3.14159)
		context.fill()		

# --------------------------------------------------------------------------------------------------------
# DEMO 3.2 : hands & skeleton detection
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_hands_and_skeleton(interpreterModel, interpreterModel_Sub, frame, PicturePath, IsLoadPicture):

	# Capture Image
	if IsLoadPicture==True :
		FrameInput  = cv2.imread(PicturePath)
	else:
		FrameInput  = frame

	# fix parameter
	IoU = 0.6
	adj_scale = 0.2
	skeleton_point = 21
	handscore = 0.25
	
	# --------------------------------------------------------------------------------------------------------
	# Hand Detection
	# --------------------------------------------------------------------------------------------------------
	# Inference
	output_Info       = RunInferenceMode(interpreterModel, FrameInput)
	output_details    = output_Info[0] ; InfereneTime = output_Info[1]
	detection_boxes   = interpreterModel.get_tensor(output_details[0]['index'])
	detection_classes = interpreterModel.get_tensor(output_details[1]['index'])
	detection_scores  = interpreterModel.get_tensor(output_details[2]['index'])
	num_boxes         = interpreterModel.get_tensor(output_details[3]['index'])
	boxs              = np.squeeze(detection_boxes)
	scores            = np.squeeze(detection_scores)
	boxs_nms, scores_nms = nms(boxs, scores, float(IoU))

	# Hands 
	output_details_Hand = []
	output_details_Skeleton  = []
	for i in range( 0, len(scores_nms)-1) :
		if scores_nms[i] > handscore:
			# 擴大偵測視窗 
			x = boxs_nms[i, [1, 3]] * frame.shape[1]
			y = boxs_nms[i, [0, 2]] * frame.shape[0]
			w = int(x[1]-x[0])
			h = int(y[1]-y[0])
			y[0] = y[0] - int(h*0.25) #[WPI] modify 0.3 to 0.35
			y[1] = y[1] - int(h*0.2) #[WPI] modify 0.3 to 0.35
			x[0] = x[0] - int(w*0.25) 
			x[1] = x[1] + int(w*0.25) 
			
			output_details_Hand.append([x,y])

			# --------------------------------------------------------------------------------------------------------
			# Skeleton Detection
			# --------------------------------------------------------------------------------------------------------
			if (output_details_Hand) :
				# Inference
				roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
				roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
				roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
				roi_y1 = min(frame.shape[1], np.floor(y[1] + 0.5).astype('int32'))
				hand_img = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1]#hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
				hand_input_data = (hand_img.astype("float32")/255)
				Skeleton_input_width    = interpreterModel_Sub.get_input_details()[0]['shape'][2]
				Skeleton_input_height   = interpreterModel_Sub.get_input_details()[0]['shape'][1]
				Skeleton_output         = RunInferenceMode(interpreterModel_Sub, hand_input_data)
				Skeleton_output_details = Skeleton_output[0] ; InfereneTime = InfereneTime + Skeleton_output[1]
				skeletonFeature         = interpreterModel_Sub.get_tensor(Skeleton_output_details[2]['index'])[0].reshape(21, 3)
				skeletonDetected        = interpreterModel_Sub.get_tensor(Skeleton_output_details[0]['index'])[0]

				# output 
				Px = []
				Py = []
				size_rate = [ hand_img.shape[1]/Skeleton_input_width, hand_img.shape[0]/Skeleton_input_height ]
				for pt in skeletonFeature:
					x = roi_x0 + int(pt[0]*size_rate[0])
					y = roi_y0 + int(pt[1]*size_rate[1])
					Px.append(x)
					Py.append(y)
				
				output_details_Skeleton.append([skeletonDetected, Px, Py])
				i = scores_nms # [WPI] Find out and stop

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [FrameOutput,[output_details_Hand, output_details_Skeleton], InfereneTime]
def draw_hands_and_skeleton(context, frame, drawout_details):

	# Define 
	hand_rect_color = (0, 255, 0)
	rect_linewidth      = 4
	label_text_fontsize = 24

	# Draw Hand
	Hand_coordinate  = drawout_details[0]
	Skeleton_details = drawout_details[1]
	for index , hand in enumerate( Hand_coordinate ):
		# 手骨偵測
		skeleton = Skeleton_details[index]
		Px       = skeleton[1]
		Py       = skeleton[2]
		if skeleton[index][0] > 0.5  :
			# 畫框
			x = hand[0]
			y = hand[1]
			w = int(x[1]-x[0])
			h = int(y[1]-y[0])
			DrawObjRect_cairo(context, int(x[0]),  int(y[0]), w, h, hand_rect_color, rect_linewidth)
			DrawObjLabelText_cairo(context, int(x[0]),  int(y[0]) , hand_rect_color, "Hand", label_text_fontsize)

			# 拇指
			context.set_source_rgb(0,1,0)
			context.set_line_width(3)
			context.move_to(Px[0], Py[0]); context.line_to(Px[1], Py[1]);  context.stroke()
			context.move_to(Px[1], Py[1]); context.line_to(Px[2], Py[2]);  context.stroke()
			context.move_to(Px[2], Py[2]); context.line_to(Px[3], Py[3]);  context.stroke()
			context.move_to(Px[3], Py[3]); context.line_to(Px[4], Py[4]);  context.stroke()
			# 食指
			context.move_to(Px[0], Py[0]); context.line_to(Px[5], Py[5]);  context.stroke()
			context.move_to(Px[5], Py[5]); context.line_to(Px[6], Py[6]);  context.stroke()
			context.move_to(Px[6], Py[6]); context.line_to(Px[7], Py[7]);  context.stroke()
			context.move_to(Px[7], Py[7]); context.line_to(Px[8], Py[8]);  context.stroke()
			# 中指
			context.move_to(Px[5], Py[5]);   context.line_to(Px[9], Py[9]);  context.stroke()
			context.move_to(Px[9], Py[9]);   context.line_to(Px[10], Py[10]);  context.stroke()
			context.move_to(Px[10], Py[10]); context.line_to(Px[11], Py[11]);  context.stroke()
			context.move_to(Px[11], Py[11]); context.line_to(Px[12], Py[12]);  context.stroke()
			# 無名指
			context.move_to(Px[9], Py[9]);   context.line_to(Px[13], Py[13]);  context.stroke()
			context.move_to(Px[13], Py[13]); context.line_to(Px[14], Py[14]);  context.stroke()
			context.move_to(Px[14], Py[14]); context.line_to(Px[15], Py[15]);  context.stroke()
			context.move_to(Px[15], Py[15]); context.line_to(Px[16], Py[16]);  context.stroke()
			# 小指
			context.move_to(Px[13], Py[13]); context.line_to(Px[17], Py[17]);  context.stroke()
			context.move_to(Px[17], Py[17]); context.line_to(Px[18], Py[18]);  context.stroke()
			context.move_to(Px[18], Py[18]); context.line_to(Px[19], Py[19]);  context.stroke()
			context.move_to(Px[19], Py[19]); context.line_to(Px[20], Py[20]);  context.stroke()
			context.move_to(Px[17], Py[17]); context.line_to(Px[0], Py[0]);  context.stroke()
			#指節
			#for i in range(len(Px)):
			#	cv2.circle(frame, ( Px[i] , Py[i] ), 1, (0, 0, 255), 4)
			#context.set_source_rgb(1,0,0)
			#context.arc( Px[i] , Py[i], 2, 0, 2 * 3.14159)
			#context.fill()	


# --------------------------------------------------------------------------------------------------------
# DEMO 3.3 : Plam and Landmark detection
# * Code Ver : 1.0
# * Code Date: 2024/12/13
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def getPalmAnchors(anchors_path): # By WPI 2024/12/12
    with open(anchors_path, "r") as csv_f:
        anchors = np.r_[[x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]]
    return anchors
palm_anchors = getPalmAnchors("../data/anchors.csv")
def detect_palm_and_landmark(interpreterModel, interpreterModel_Sub, frame, PicturePath, IsLoadPicture):
	# Capture Image
	if IsLoadPicture==True :
		FrameInput  = cv2.imread(PicturePath)
	else:
		FrameInput  = frame

	FrameInput_BGR      = cv2.cvtColor(FrameInput, cv2.COLOR_BGR2RGB)

	# Pre-Process
	shape = np.r_[FrameInput.shape]
	pad = (shape.max() - shape[:2]).astype('uint32') // 2
	frame_pad = np.pad(FrameInput_BGR,((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),mode='constant')
	frame_resized = cv2.resize(frame_pad, (256, 256))
	frame_resized = np.ascontiguousarray(frame_resized)
	frame_resized_norm = ((frame_resized / 127.5) - 1.0).astype('float32')

	# --------------------------------------------------------------------------------------------------------
	# Hand Detection
	# --------------------------------------------------------------------------------------------------------
	# Inference
	output_Info_hand    = RunInferenceMode(interpreterModel, frame_resized_norm) #interpreterModel = interpreterPalmDetect
	output_details_hand = output_Info_hand[0] ; InfereneTime = output_Info_hand[1]
	
	# get hand feature
	detection_boxes   = interpreterModel.get_tensor(output_details_hand[1]['index'])[0]#out_reg_idx
	detection_scores = interpreterModel.get_tensor(output_details_hand[0]['index'])[0,:,0]
	detection_scores = np.clip(detection_scores, -20, 20)
	detecion_mask = (1 / (1 + np.exp(-detection_scores)))> 0.95
	candidate_detect = detection_boxes[detecion_mask]
	candidate_anchors = palm_anchors[detecion_mask]
	
	# calcuate key point
	kp_orig = []
	if candidate_detect.shape[0] != 0:
		max_idx = np.argmax(candidate_detect[:, 3])
		dx,dy,w,h = candidate_detect[max_idx, :4]
		center_wo_offst = candidate_anchors[max_idx,:2] * 256
		keypoints = center_wo_offst + candidate_detect[max_idx,4:].reshape(-1,2)

		kp_tmp = keypoints[2] - keypoints[0]
		kp_tmp /= np.linalg.norm(kp_tmp)
		kp_tmp_r = kp_tmp @ np.r_[[[0,1],[-1,0]]].T
		source = np.float32([keypoints[2], keypoints[2]+kp_tmp*max(w,h) * 1.5, keypoints[2] + kp_tmp_r*max(w,h) * 1.5])
		source -= (keypoints[0] - keypoints[2]) * 0.2 

        # ----------------------------------------------------------------------------------------------------
        # Landmark By WPI 2024/12/12
        # ----------------------------------------------------------------------------------------------------
        # warpAffine (slow)
		Mtr = cv2.getAffineTransform(source * max(FrameInput_BGR.shape) / 256, np.float32([[128, 128],[128,   0],[  0, 128]]))
		img_landmark = cv2.warpAffine(((frame_pad.astype(np.float32) / 127.5) - 1.0), Mtr, (256,256), flags=cv2.INTER_NEAREST)


		# get landmark feature
		output_Info_landmark    = RunInferenceMode(interpreterModel_Sub, img_landmark)
		output_details_landmark = output_Info_landmark[0] ; 
		joints = interpreterModel_Sub.get_tensor(output_details_landmark[0]['index'])[0]	
		joints = joints.tolist() ; del joints[2::3]
		joints = np.array(joints)
		joints = joints.reshape(-1,2)
		
		Mtr = np.pad(Mtr.T, ((0,0),(0,1)), constant_values=1, mode='constant').T
		Mtr[2,:2] = 0 ; Minv = np.linalg.inv(Mtr)
		kp_orig = (np.pad(joints, ((0,0),(0,1)), constant_values=1, mode='constant') @ Minv.T)[:,:2]
		kp_orig -= pad[::-1]

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [FrameOutput,[kp_orig], InfereneTime]
def draw_palm_and_landmark(context, frame, drawout_details):
	points = drawout_details[0]
	connections = [
		(5, 6), (6, 7), (7, 8),
		(9, 10), (10, 11), (11, 12),
		(13, 14), (14, 15), (15, 16),
		(0, 5), (5, 9), (9, 13), (13, 17), (0, 9), (0, 13)]
	connections += [(0, 17), (17, 18), (18, 19), (19, 20), (0, 1), (1, 2), (2, 3), (3, 4)]
	if points is not None:
		context.set_source_rgb(0,1,0)
		context.set_line_width(3)
		for connection in connections:
			x0, y0 = points[connection[0]]
			x1, y1 = points[connection[1]]
			context.move_to(int(x0), int(y0)); context.line_to(int(x1), int(y1));  context.stroke()#cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

# --------------------------------------------------------------------------------------------------------
# DEMO 3.4 : pose detection (YOLOv8s)
# * Code Ver : 2.0
# * Code Date: 2024/11/25
# * Author   : Weilly Li
# * Update   : Modify y (output) after Tensorflow Lite 
# --------------------------------------------------------------------------------------------------------
def detect_posedetect_YOLOv8s(interpreterModel, frame, PicturePath, IsLoadPicture):

	# Param Setting
	obj_threshold  =0.1
	IoU = 0.45
	detMax = 300
	
	# Capture Image
	if IsLoadPicture==True :
		FrameInput  = cv2.imread(PicturePath)
	else:
		FrameInput  = frame

	# Inference
	width          = interpreterModel.get_input_details()[0]['shape'][2]
	height         = interpreterModel.get_input_details()[0]['shape'][1]
	input_details  = interpreterModel.get_input_details()
	output_Info    = RunInferenceMode_YOLO(interpreterModel, FrameInput ) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# Inference of Ouput
	y = PostPrecess_YOLO_Ouput_tflite_V2(interpreterModel, output_details, True)

	# NMS 
	preds = non_max_suppression_YOLOv8(torch.from_numpy(y[0]), obj_threshold, IoU, agnostic=False, max_det=detMax, classes=None, nc=1)
	
	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [ FrameOutput, [[width,height] , preds], InfereneTime]
def draw_posedetect_YOLOv8s(context, frame, drawout_details):
	# DrawOut
	preds  = drawout_details[1]
	width  = drawout_details[0][0]
	height = drawout_details[0][1]
	
	# draw
	frame_shape = (frame.shape[0], frame.shape[1])
	annotator = Annotator(frame, line_width=3)

	# Plot Detect results
	PostPrecess_YOLO_Plot_Box(context, preds, annotator, coco_dict, [drawout_details[0][0],drawout_details[0][1]], frame.shape)

	# Plot Pose results
	PostPrecess_YOLO_Plot_Pose(context, preds, annotator, [drawout_details[0][0],drawout_details[0][1]], frame.shape)

# --------------------------------------------------------------------------------------------------------
# DEMO TBC : body / Pose detection (MobilNet)
# * Code Ver : 1.0
# * Code Date: 2023/07/17
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_posedetect_mobilenetssd(interpreterModel, frame, PicturePath, IsLoadPicture):

	# Capture Image
	if IsLoadPicture==True :
		# Loading Image
		FrameInput = cv2.imread(PicturePath)
		FrameInputRGB  = cv2.cvtColor(FrameInput, cv2.COLOR_BGR2RGB)
	else:
		# Camera Streamer
		FrameInput     = frame
		FrameInputRGB  = frame

	# --------------------------------------------------------------------------------------------------------
	# Body Detection
	# --------------------------------------------------------------------------------------------------------
	# get interpreter result
	output_Info       = RunInferenceMode(interpreterModel, FrameInputRGB )
	output_details    = output_Info[0] ; InfereneTime = output_Info[1]
	heat_maps         = interpreterModel.get_tensor(output_details[0]['index'])
	offset_maps       = interpreterModel.get_tensor(output_details[1]['index'])

	# build output image
	height_         = heat_maps.shape[1]
	width_          = heat_maps.shape[2]
	num_key_points  = heat_maps.shape[3]
	key_point_positions = [[0] * 2 for i in range(num_key_points)]
	for key_point in range(num_key_points):
		max_val = heat_maps[0][0][0][key_point]
		max_row = 0
		max_col = 0
		for row in range(height_):
			for col in range(width_):
				if heat_maps[0][row][col][key_point] > max_val:
					max_val = heat_maps[0][row][col][key_point]
					max_row = row
					max_col = col
				key_point_positions[key_point] = [max_row, max_col]
				
	# scale key point
	x_coords = [0] * num_key_points
	y_coords = [0] * num_key_points
	confidence_scores = [0] * num_key_points
				
	for i, position in enumerate(key_point_positions):
		position_y = int(key_point_positions[i][0])
		position_x = int(key_point_positions[i][1])
		y_coords[i] = int(position[0])
		x_coords[i] = int(position[1])
		confidence_scores[i] = (float)(heat_maps[0][position_y][position_x][i] /255)
				
	# body score
	person = Person()
	key_point_list = []
	total_score = 0
	for i in range(num_key_points):
			key_point = KeyPoint()
			key_point_list.append(key_point)

	for i, body_part in enumerate(BodyPart):
		key_point_list[i].bodyPart = body_part
		key_point_list[i].position.x = x_coords[i]
		key_point_list[i].position.y = y_coords[i]
		key_point_list[i].score = confidence_scores[i]
		total_score += confidence_scores[i]
				
	# build - body label
	person.keyPoints = key_point_list
	person.score = total_score / num_key_points
	body_joints = [[BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW],
					[BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER],
					[BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER],
					[BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW],
					[BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST],
					[BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP],
					[BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP],
					[BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER],
					[BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE],
					[BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE],
					[BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE],
					[BodyPart.RIGHT_KNEE,BodyPart.RIGHT_ANKLE]]

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [FrameOutput, [[width_, height_], [person,position,body_joints]], InfereneTime]
def draw_posedetect_mobilenetssd(context, frame, drawout_details):
	width_  = drawout_details[0][0]
	height_ = drawout_details[0][1]
	person  = drawout_details[1][0]
	position= drawout_details[1][1]
	body_joints = drawout_details[1][2]
	context.set_source_rgb(0,1,0)
	context.set_line_width(3)

	# draw - body
	for line in body_joints:
		if person.keyPoints[line[0].value[0]].score > 0.35 and person.keyPoints[line[1].value[0]].score > 0.35:
			start_point_x = (int)(person.keyPoints[line[0].value[0]].position.x  * frame.shape[1]/width_)
			start_point_y = (int)(person.keyPoints[line[0].value[0]].position.y  * frame.shape[0]/height_ )
			end_point_x   = (int)(person.keyPoints[line[1].value[0]].position.x  * frame.shape[1]/width_)
			end_point_y   = (int)(person.keyPoints[line[1].value[0]].position.y  * frame.shape[0]/height_ )
			#cv2.line(frame, (start_point_x, start_point_y) , (end_point_x, end_point_y), (255, 255, 0), 3)
			context.move_to(start_point_x, start_point_y); context.line_to(end_point_x, end_point_y);  context.stroke()
					
	# draw - head
	left_ear_x   = (int)(person.keyPoints[3].position.x  * frame.shape[1]/width_)
	left_ear_y   = (int)(person.keyPoints[3].position.y  * frame.shape[0]/height_)
	right_ear_x  = (int)(person.keyPoints[4].position.x  * frame.shape[1]/width_)
	right_ear_y  = (int)(person.keyPoints[4].position.y  * frame.shape[0]/height_)
	left_shoulder_x   = (int)(person.keyPoints[5].position.x  * frame.shape[1]/width_)
	left_shoulder_y   = (int)(person.keyPoints[5].position.y  * frame.shape[0]/height_)
	right_shoulder_x  = (int)(person.keyPoints[6].position.x  * frame.shape[1]/width_)
	right_shoulder_y  = (int)(person.keyPoints[6].position.y  * frame.shape[0]/height_)
	start_point_x = (int) ((left_ear_x + right_ear_x)/2 )
	start_point_y = left_ear_y
	if(right_ear_y < left_ear_y) : start_point_y = right_ear_y 
	end_point_x = (int) ((left_shoulder_x + right_shoulder_x)/2 )
	end_point_y = left_shoulder_y
	if(right_shoulder_y > left_shoulder_y) : end_point_y = right_shoulder_y
	#print(start_point_x, start_point_y)
	context.move_to(start_point_x, start_point_y); context.line_to(end_point_x, end_point_y);  context.stroke()



# --------------------------------------------------------------------------------------------------------
# ------------------------------------------   (4) Other   ----------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# DEMO 4.1 : Age, Gender, Enthnicity detection
# * Code Ver : 3.0
# * Code Date: 2023/09/08
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_age_gender_recognition(interpreterModel, interpreterModel_Sub, interpreterModel_Sub2, frame, PicturePath, IsLoadPicture):

	# Capture Image
	if IsLoadPicture==True :
		FrameInput  = cv2.imread("/home/root/ATU-Camera-GTK/data/img/DataBase_Emtion/Rainine.jpg")
	else:
		FrameInput = frame

	# Info
	IoU                = 0.5
	adj_x_scale        = 0.1
	adj_y_scale        = 0.05
	offset_y           = 5
	obj_threhsold      = 0.6

	# --------------------------------------------------------------------------------------------------------
	# Face Detection
	# --------------------------------------------------------------------------------------------------------
	frameGray         = cv2.cvtColor(FrameInput, cv2.COLOR_BGR2GRAY) 

	# Inference 
	output_Info       = RunInferenceMode(interpreterModel, FrameInput) 
	output_details    = output_Info[0] ; InfereneTime = output_Info[1] 
	detection_boxes   = interpreterModel.get_tensor(output_details[0]['index'])
	detection_classes = interpreterModel.get_tensor(output_details[1]['index'])
	detection_scores  = interpreterModel.get_tensor(output_details[2]['index'])
	num_boxes         = interpreterModel.get_tensor(output_details[3]['index'])
	boxs              = np.squeeze(detection_boxes)
	scores            = np.squeeze(detection_scores) #(10,)
	boxs_nms, scores_nms = nms(boxs, scores, float(IoU))

	# Outputs
	output_details_Face = []
	output_details_AgeGender  = []
	output_details_Emotion  = []
	for i in range( 0, len(scores_nms)-1) :
		if scores_nms[i] > obj_threhsold: 

			#  Face
			x = boxs_nms[i, [1, 3]] * frame.shape[1]
			y = boxs_nms[i, [0, 2]] * frame.shape[0]
			x[1] = x[1] + int((x[1]-x[0])*adj_x_scale)
			y[1] = y[1] + int((y[1]-y[0])*adj_y_scale) 
			output_details_Face.append([x,y])

			# --------------------------------------------------------------------------------------------------------
			# Age-Gender Recognition
			# --------------------------------------------------------------------------------------------------------
			# Face Image
			roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
			roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
			roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
			roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))
			face_img                 = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1]

			# Inference
			AgeGender_input_width    = interpreterModel_Sub.get_input_details()[0]['shape'][2]
			AgeGender_input_height   = interpreterModel_Sub.get_input_details()[0]['shape'][1]
			AgeGender_output         = RunInferenceMode(interpreterModel_Sub, face_img)
			AgeGender_output_details = AgeGender_output[0]; InfereneTime = InfereneTime + AgeGender_output[1] 
			Age                      = interpreterModel_Sub.get_tensor(AgeGender_output_details[0]['index'])*100
			Gender                   = gender_dict[np.argmax(interpreterModel_Sub.get_tensor(AgeGender_output_details[1]['index']))]
			output_details_AgeGender.append([Age,Gender])

			# --------------------------------------------------------------------------------------------------------
			# Emotion Recognition
			# --------------------------------------------------------------------------------------------------------
			# Face Image(Gray)
			face_grey                = frameGray[ roi_y0 : roi_y1, roi_x0 : roi_x1] 

			# Inference
			Emotion_input_width    = interpreterModel_Sub2.get_input_details()[0]['shape'][2]
			Emotion_input_height   = interpreterModel_Sub2.get_input_details()[0]['shape'][1]
			Emotion_output         = RunInferenceMode(interpreterModel_Sub2, face_grey)
			Emotion_output_details = Emotion_output[0]; InfereneTime = InfereneTime + Emotion_output[1] 
			EmotionPredict         = interpreterModel_Sub2.get_tensor(Emotion_output_details[0]['index'])
			output_details_Emotion.append(EmotionPredict)

	# Destination of Buffer
	FrameOutput = GetDstBuffer(FrameInput, frame.shape[1], frame.shape[0], IsLoadPicture)
	return [FrameOutput, [output_details_Face, output_details_AgeGender, output_details_Emotion], InfereneTime]
def draw_age_gender_recognition(context, frame, drawout_details):

	# define
	face_rect_color    = (0, 255, 0)
	face_label_color   = (200, 160, 80)
	face_rect_width    = 4
	face_circle_radius = 2
	label_text_fontsize= 2
	label_text_fix_x   = 5
	label_text_fix_y   = 35

	# Draw Face
	faces = drawout_details[0]
	AgeGender = drawout_details[1]
	Emotion   = drawout_details[2]
	faceN = 0
	for face in faces :
		x = face[0]
		y = face[1]
		w = int(x[1]-x[0])
		h = int(y[1]-y[0])
		DrawObjRect_cairo(context, int(x[0]),  int(y[0]), w, h, face_rect_color, face_rect_width)#DrawObjLabelText_cairo(context, int(x[0]),  int(y[0]) , face_rect_color, "face", label_text_fontsize)
		DrawObjLabelText_cairo(context, int(x[0]),  int(y[0])  , face_rect_color, "Age : "+  str(AgeGender[faceN][0][0][0][0][0])[:3] + "Gender : " + AgeGender[faceN][1], 1)
		DrawObjLabelText_cairo(context, int(x[0]) + label_text_fix_x ,  int(y[0]) + label_text_fix_y  , face_label_color, emotion_dict[np.argmax(Emotion)], 1)
		faceN = faceN + 1 


# --------------------------------------------------------------------------------------------------------
# DEMO 4.2 : Face Recognition
# * Code Ver : 2.0
# * Code Date: 2023/08/31
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def load_faceRecognitionDataBase(interpreterModel, DataBase_Path, List_Feature, List_Image):
	#database_folder = "../data/img/DataBase_Face_Recognition/"
	folder = DataBase_Path
	List_PersonalName = [os.path.join(folder, f) for f in os.listdir(folder)]
	input_details  = interpreterModel.get_input_details()
	output_details = interpreterModel.get_output_details()
	width          = input_details[0]['shape'][2]
	height         = input_details[0]['shape'][1]
	for data_path in List_PersonalName :
		img  = cv2.imread(data_path) ; List_Image.append(img)
		img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		img  = cv2.resize(img, (width, height))
		img  = np.expand_dims(img, axis=0).astype("float32")
		interpreterModel.set_tensor(input_details[0]['index'], img)  # 先行進行暖開機
		interpreterModel.invoke()
		List_Feature.append( interpreterModel.get_tensor(output_details[0]['index']) )
	return List_PersonalName
def detect_faceRecognition(interpreterModel, interpreterModel_Sub, List_PersonalName , frame, List_Feature, List_Image, IsTouchSignal):
	
	# Info
	IoU                   = 0.4
	adj_x_scale           = 0.1
	adj_y_scale           = 0.05
	offset_y              = 5
	obj_threhsold         = 0.5
	index_similarity_base = 15
	default_database_name = 8 # import

	# --------------------------------------------------------------------------------------------------------
	# Face Detection
	# --------------------------------------------------------------------------------------------------------
	# Inference
	output_Info       = RunInferenceMode(interpreterModel, frame) 
	output_details    = output_Info[0] ; InfereneTime = output_Info[1]
	detection_boxes   = interpreterModel.get_tensor(output_details[0]['index'])
	detection_classes = interpreterModel.get_tensor(output_details[1]['index'])
	detection_scores  = interpreterModel.get_tensor(output_details[2]['index'])
	num_boxes         = interpreterModel.get_tensor(output_details[3]['index'])
	boxs              = np.squeeze(detection_boxes)
	scores            = np.squeeze(detection_scores)
	boxs_nms, scores_nms = nms(boxs, scores, float(IoU))

	# output 
	output_details_FaceInfo = []
	output_details_FaceRecognition  = []

	if scores_nms[0] > obj_threhsold : 
		
		#  Face ROI Setting
		x = boxs_nms[0, [1, 3]] * frame.shape[1]
		y = boxs_nms[0, [0, 2]] * frame.shape[0]
		x[0] = x[0] - 5 
		x[1] = x[1] + 5
		output_details_FaceInfo.append([x,y])

		# --------------------------------------------------------------------------------------------------------
		# Face Recognition
		# --------------------------------------------------------------------------------------------------------
		# Get Face Buffer
		roi_x0          = max(0, np.floor(x[0] + 0.5).astype('int32'))
		roi_y0          = max(0, np.floor(y[0] + 0.5).astype('int32'))
		roi_x1          = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
		roi_y1          = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))
		face_img        = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1]
		face_img        = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
		face_img        = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
		face_input_data = face_img.astype("float32")
		# Get Inference Info
		FaceRecognition_input_width    = interpreterModel_Sub.get_input_details()[0]['shape'][2]
		FaceRecognition_input_height   = interpreterModel_Sub.get_input_details()[0]['shape'][1]
		FaceRecognition_output         = RunInferenceMode(interpreterModel_Sub, face_input_data)
		FaceRecognition_output_details = FaceRecognition_output[0] ; InfereneTime = InfereneTime + FaceRecognition_output[1]
		faceFeaturePoint               = interpreterModel_Sub.get_tensor(FaceRecognition_output_details[0]['index'])
		
		# similarity
		similarity = []
		for Feature in List_Feature :
			similarity.append(cos_similarity(faceFeaturePoint[0], Feature[0]))
		similarity_max   = np.max(similarity)
		similarity_idx   = similarity.index(similarity_max)
		similarity_name  = List_PersonalName[similarity_idx]
		similarity_name_ = List_PersonalName[similarity_idx][List_PersonalName[similarity_idx].find("DataBase_Face_Recognition/")+8:List_PersonalName[similarity_idx].find("-")]

		# Rigester
		face_saveSingal  = IsTouchSignal[0]
		face_touch_x     = int(IsTouchSignal[1]) 
		face_touch_y     = int(IsTouchSignal[2]) 
		if (face_saveSingal and
			face_touch_x > roi_x0 and face_touch_x < roi_x1 and
		    face_touch_y > roi_y0 and face_touch_y < roi_y1 ):
			print("Register Test"  + str(len(List_PersonalName) - default_database_name) )
			List_PersonalName.append("../data/img/DataBase_Face_Recognition/TEST_" + str(len(List_PersonalName) - default_database_name) + "-")
			List_Feature.append(faceFeaturePoint)
			List_Image.append(frame[ roi_y0 : roi_y1, roi_x0 : roi_x1])
			Singal_Register = False
			
		# 輸出
		output_details_FaceRecognition.append([similarity_name, similarity_max])

	return [frame, [output_details_FaceInfo, output_details_FaceRecognition], InfereneTime]
def draw_faceRecognition(context, frame, drawout_details, frameSamplePath):
	
	# define
	face_rect_color       = (0, 255, 0)
	face_label_green      = (100, 180, 0)
	info__text_red        = (255,0,0)
	rect_linewidth        = 4
	label_text_fontsize   = 2
	label_text_fix_y      = 5
	recognition_threshold = 0.95
	faces                 = drawout_details[0]
	facerecognition       = drawout_details[1]

	# face
	for face in faces :
		x = face[0]
		y = face[1]
		w = int(x[1]-x[0])
		h = int(y[1]-y[0])
		DrawObjRect_cairo(context, int(x[0]),  int(y[0]), w, h, face_rect_color, rect_linewidth)
		#DrawObjLabelText_cairo(context, int(x[0]),  int(y[0]) , face_rect_color, "face", label_text_fontsize)

	#Recognition Info
	for info in facerecognition :
		similarity_value= info[1]
		similarity_name = info[0][info[0].find("_Face_Recognition/")+18:info[0].find("-")]
		coordinate_x    = int(x[0])
		coordinate_y    = int(y[0]) - label_text_fix_y
		if similarity_value > recognition_threshold : 
			DrawObjLabelText_cairo(context, coordinate_x, coordinate_y, face_rect_color,  similarity_name + " , (%) : " + str(similarity_value)[:4] , label_text_fontsize)
		else :
			DrawObjLabelText_cairo(context, coordinate_x, coordinate_y, face_rect_color,  "Nobody", label_text_fontsize)
	# Print Sample Image
	#context.set_source_surface( cairo.ImageSurface.create_from_png(frameSamplePath) ,0,0)
	#context.paint()


# --------------------------------------------------------------------------------------------------------
# DEMO 4.3 : Depth Estimation (MegaDepth)
# * Code Ver : 1.0
# * Code Date: 2023/09/12
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------
def detect_depthestimation_MegaDepth(interpreterModel, frame, PicturePath, IsLoadPicture):
	# Capture Image
	if IsLoadPicture==True :
		# Loading Image
		FrameInput    = cv2.imread(PicturePath)
		FrameInputRGB = cv2.cvtColor(FrameInput, cv2.COLOR_BGR2RGB) 
	else:
		# Camera Streamer
		FrameInput = frame
		FrameInputRGB = cv2.cvtColor(FrameInput, cv2.COLOR_BGR2RGB) 

	# Inference
	output_Info    = RunInferenceMode(interpreterModel,FrameInputRGB) 
	output_details = output_Info[0] ; InfereneTime = output_Info[1] 

	# depth
	depth = interpreterModel.get_tensor(output_details[0]['index'])[0]
	depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
	
	# Destination of Buffer
	FrameOutput = GetDstBuffer(depth_rgb, frame.shape[1], frame.shape[0], 1)
	return [FrameOutput, 1, InfereneTime]
def draw_depthestimation_MegaDepth(context, frame, drawout_details):
	# Nothing
	return 0
		