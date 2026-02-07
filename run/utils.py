# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2023 Freescale Semiconductor
# Copyright 2023 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 3.0
# * Code Date: 2023/07/14
# * Author   : Weilly Li
# * update : add ethous delegate 
#--------------------------------------------------------------------------------------

import re
import cv2
import gi
import time
import colorsys
import random
import torch
import torchvision
import numpy as np
import tflite_runtime.interpreter as tflite
from enum import Enum

gi.require_version("Gtk", "3.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gtk, Gst, Gio, GLib, Gdk

#-------------------------------------------------------------------------------------------------------------------
# GTK GUI Func
#-------------------------------------------------------------------------------------------------------------------
def create_button_with_icon(icon_name, style_class, callback, header=None):
    button = Gtk.Button()
    icon = Gio.ThemedIcon(name=icon_name)
    image = Gtk.Image.new_from_gicon(icon, Gtk.IconSize.BUTTON)
    button.add(image)
    button.get_style_context().add_class(style_class)
    button.connect("clicked", callback)
    if header:
        header.pack_start(button)
    return button


#-------------------------------------------------------------------------------------------------------------------
# Generate Model Status
#-------------------------------------------------------------------------------------------------------------------
def writeFlagInference(Texts):
    f = open("../data/tmp/flaginference.txt", 'w')
    f.write(Texts);f.close()

def getFlagInference():
    f = open("../data/tmp/flaginference.txt", 'r', encoding='utf-8')
    flag_ = f.read()
    f.close()
    return flag_

def writeExtraLoadingPictue(Texts):
    f = open("../data/tmp/extra_loadingPicture.txt", 'w')
    f.write(Texts);f.close()

def getExtraLoadingPictue():
    f = open("../data/tmp/extra_loadingPicture.txt", 'r', encoding='utf-8')
    status = f.read()
    f.close()
    return status

def writeModelStatus(Texts):
    f = open("../data/tmp/modelstate.txt", 'w')
    f.write(Texts);f.close()

def getModelStatus():
    f = open("../data/tmp/modelstate.txt", 'r', encoding='utf-8')
    modelstatus = f.read()
    f.close()
    return modelstatus

def writeFilePath(Texts):
    f = open("../data/tmp/filepath.txt", 'w')
    f.write(Texts);f.close()

def getFilePath():
    f = open("../data/tmp/filepath.txt", 'r', encoding='utf-8')
    filepath = f.read()
    f.close()
    return filepath

def getCameraDeviceStatus():
    f = open("CameraDevice.txt", 'r', encoding='utf-8')
    cameradevice = f.read()
    f.close()
    if (len(cameradevice)<=2):
        return int(cameradevice)
    else:
        return str(cameradevice)

#-------------------------------------------------------------------------------------------------------------------
# Tensorflow Interprter
#-------------------------------------------------------------------------------------------------------------------

def InferenceDelegate( model, delegate ):
    if (delegate=="vx") :
        interpreter = tflite.Interpreter(model, experimental_delegates=[tflite.load_delegate("/usr/lib/libvx_delegate.so")])
    elif(delegate=="ethosu"):
        interpreter = tflite.Interpreter(model, experimental_delegates=[tflite.load_delegate("/usr/lib/libethosu_delegate.so")])
    elif(delegate=="neutrons"):
        interpreter = tflite.Interpreter(model, experimental_delegates=[tflite.load_delegate("/usr/lib/libneutron_delegate.so")])
    elif(delegate=="xnnpack"):
        interpreter = tflite.Interpreter(model)
    else :
        print("ERROR : Deleget Input Fault")
        return 0
    return interpreter

def GetInferenceMode(model, delegate):
    # Generator
    interpreter = InferenceDelegate(model, delegate)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width    = input_details[0]['shape'][2]
    height   = input_details[0]['shape'][1]
    nChannel = input_details[0]['shape'][3]

    # Warm-up
    if(input_details[0]['dtype']==np.uint8):
        interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("uint8") )
    else :
        interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("float32") )
    interpreter.invoke()
    print( model , " loading done!! " )
    print( "----------------------------------------------------------------------------" )
    return interpreter

def RunInferenceMode(model, frame):
    # get interpreter info
    input_details  = model.get_input_details()
    output_details = model.get_output_details()
    width          = input_details[0]['shape'][2]
    height         = input_details[0]['shape'][1]
    nChannel       = input_details[0]['shape'][3]
    isONNX          = False # input process
 
    if (height==3) :#isONNX
        width    = input_details[0]['shape'][3]
        height   = input_details[0]['shape'][2]
        nChannel = input_details[0]['shape'][1]
        isONNX   = True

    # data
    frame_resized = cv2.resize(frame, (width, height))
    try:
        isRGB      = frame_resized.shape[2]
        input_data = np.expand_dims(frame_resized, axis=0)
        if(isONNX):
            input_data = np.swapaxes(input_data, 1,3)
            input_data = np.swapaxes(input_data, 2,3)

    except Exception :
        input_data = np.expand_dims(frame_resized, axis=0)
        input_data = np.expand_dims(input_data, axis=3)

    # data expand
    if (input_details[0]['dtype']==np.uint8) :
        model.set_tensor(input_details[0]['index'],input_data.astype('uint8'))
    else:
        model.set_tensor(input_details[0]['index'],input_data.astype('float32'))

    # interpreter
    time_start = time.time()
    model.invoke()
    time_end = time.time()
    inference_time = (time_end - time_start)
    
    return [output_details, inference_time]

def RunInferenceMode_YOLO(model, frame):
    # get interpreter result
    width             = model.get_input_details()[0]['shape'][2]
    height            = model.get_input_details()[0]['shape'][1]
    input_details     = model.get_input_details()
    output_details    = model.get_output_details()
    scale, zero_point = input_details[0]['quantization']

    # data
    frame_resized = letterbox(frame, (width,height), stride=32, auto=False)[0]
    frame_resized = frame_resized.transpose((2, 0, 1))[::-1] 
    frame_resized = np.ascontiguousarray(frame_resized)
    if (input_details[0]['dtype']==np.uint8) :
        frame_resized = (frame_resized/scale) + zero_point
        input_data    = np.expand_dims(frame_resized , axis=0).astype(np.uint8)
        input_data = input_data.swapaxes(1,2)
        input_data = input_data.swapaxes(2,3)
    else :
        frame_resized = (frame_resized/255.0) 
        input_data = np.expand_dims(frame_resized , axis=0).astype(np.float32) 
        input_data = input_data.swapaxes(1,2)
        input_data = input_data.swapaxes(2,3)
    model.set_tensor(input_details[0]['index'], input_data ) 

    # interpreter
    time_start = time.time()
    model.invoke()
    time_end = time.time()
    inference_time = (time_end - time_start)

    return [output_details, inference_time]

#-------------------------------------------------------------------------------------------------------------------
# Label
#-------------------------------------------------------------------------------------------------------------------

def load_labels(path):
    p = re.compile('\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
    return {int(num): text.strip() for num, text in lines}

def generate_colors(labels):
	hsv_tuples = [(x / len(labels), 1., 1.) for x in range(len(labels))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255),int(x[2] * 255)), colors))
	random.seed(10101)
	random.shuffle(colors)
	random.seed(None)
	return colors

#-------------------------------------------------------------------------------------------------------------------
# NMS
#-------------------------------------------------------------------------------------------------------------------
def nms(boxes, scores, Nt):
    if len(boxes) == 0:
        return [], []
    bboxes = np.array(boxes)

    # 計算 n 個候選窗的面積大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 進行排序 (默認從小到大排序)
    order = np.argsort(scores)

    picked_boxes   = []
    picked_scores  = []
    while order.size > 0:
        # 加入將當前最大的信心度之數值
        index = order[-1]
        picked_boxes.append(boxes[index])
        picked_scores.append(scores[index])

        # 獲得當前信心度候選窗與其他候選窗的相交面積
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < Nt)
        order = order[left]

    # 轉 numpy
    picked_boxes  = np.array(picked_boxes)
    picked_scores = np.array(picked_scores)

    return picked_boxes, picked_scores

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.6, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''

    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]
    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    return conf_keep_idx[pick]

def non_max_suppression_YOLOv5(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            #LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def non_max_suppression_YOLOv8(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

    return output

def normalization(data): #Person Segmetation
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#-------------------------------------------------------------------------------------------------------------------
# Draw
# https://blog.csdn.net/qq_44001342/article/details/120738484
#-------------------------------------------------------------------------------------------------------------------
def DrawObjRect_cairo(context, x, y, width, height, color, line_width):
	# draw out - box
	context.set_source_rgb(color[0]/255,color[1]/255,color[2]/255)
	context.set_line_width(line_width)
	context.rectangle(x, y, width, height) #double x, double y, double width, double height)
	context.stroke()

def DrawObjRectFull_cairo(context, x, y, width, height, color, line_width):
	# draw out - box
	context.set_source_rgb(color[0]/255,color[1]/255,color[2]/255)
	context.set_line_width(line_width)
	context.rectangle(x, y, width, height) #double x, double y, double width, double height)
	context.fill()
    
def DrawObjLabelText_cairo(context, x, y, color, text, font_size):
	context.set_font_size(font_size)
	context.set_source_rgb(color[0]/255,color[1]/255,color[2]/255)
	context.move_to(x, y-10)
	context.show_text(text)

def DrawObjInfoText_cairo(context, x, y, color, sentence):
    rect_left   = int(x + 3)
    rect_top    = int(y + 20)
    text_line = sentence.split("\n")
    text_size, baseline = cv2.getTextSize( str(sentence), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1 )
    context.set_font_size(24 *1)
    context.set_source_rgb(color[0]/255,color[1]/255,color[2]/255)

    for i, text in enumerate(text_line):
        if text :
            line_row = [ rect_left, rect_top + ( text_size[1] + 2 + baseline)*i ]
            context.move_to(line_row[0], line_row[1]-10)
            context.show_text(text)

#-------------------------------------------------------------------------------------------------------------------
# FaceMesh
#-------------------------------------------------------------------------------------------------------------------
# 特徵點判斷是否為臉部輪廓，若是則存進 list 結構中
def IsFaceContoursAndAppend( point_idx, point_x, point_y, list_index, list_data):
    # 外圍輪廓
    if ( point_idx==10  or point_idx==109 or point_idx==67  or point_idx==103 or point_idx==54  or point_idx==21  or point_idx==162 or
         point_idx==127 or point_idx==234 or point_idx==93  or point_idx==132 or point_idx==58  or point_idx==172 or point_idx==136 or
         point_idx==150 or point_idx==149 or point_idx==176 or point_idx==148 or point_idx==152 or point_idx==377 or point_idx==378 or
         point_idx==395 or point_idx==394 or point_idx==365 or point_idx==397 or point_idx==367 or point_idx==416 or point_idx==435 or
         point_idx==376 or point_idx==352 or point_idx==345 or point_idx==372 or point_idx==368 or point_idx==300 or point_idx==284 or
         point_idx==332 or point_idx==297 or point_idx==338)  :
         list_index.append(int(point_idx))
         list_data.append([point_x,point_y])


#-------------------------------------------------------------------------------------------------------------------
# Face Recognition
#-------------------------------------------------------------------------------------------------------------------
def cos_similarity(X, Y):
    Y = Y.T
    # (128,) x (n, 128) = (n,)
    result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
    return result

#-------------------------------------------------------------------------------------------------------------------
# FaceMask
#-------------------------------------------------------------------------------------------------------------------
def pad_input_image(img):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    img_pad_w = 0

    if img_w > img_h:
        img_pad_h = img_w - img_h
    else:
        img_pad_w = img_h - img_w

    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,cv2.BORDER_CONSTANT)
    return img

def decode_bbox(bbox, priors):
    """
    Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    """
    variances = [0.1, 0.2]

    boxes = np.concatenate( (priors[:, :2] + bbox[:, :2] * variances[0] * priors[:, 2:] ,  priors[:, 2:] * np.exp(bbox[:, 2:] * variances[1]) ), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def softmax_v(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
        #print("vector")
    return x 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

#-------------------------------------------------------------------------------------------------------------------
# body / Pose detection
#-------------------------------------------------------------------------------------------------------------------
class Person:
  def __init__(self):
      self.keyPoints = []
      self.score = 0.0

class Position:
  def __init__(self):
    self.x = 0
    self.y = 0

class BodyPart(Enum):
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16,

class KeyPoint:
  def __init__(self):
    self.bodyPart = BodyPart.NOSE
    self.position = Position()
    self.score = 0.0


#-------------------------------------------------------------------------------------------------------------------
#  YOLO series (Pose/PCB/Object/Segmeation)
#-------------------------------------------------------------------------------------------------------------------
FONT = 'Arial.ttf' 
class Colors_:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors_ = Colors_()  # create instance for 'from utils.plots import colors'
kpt_color = colors_.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        self.limb_color = colors_.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.skeleton   = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                          [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.kpt_color  = colors_.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def box_label(self, context, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height (WARNING: deprecated) in 9.2.0
                # _, _, w, h = self.font.getbbox(label)  # text width, height (New)
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  
            text_fontsize = 24
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            w = p2[0]-p1[0]
            h = p2[1]-p1[1]
            label_color = [color[2],color[1],color[0]]
            DrawObjRect_cairo(context, p1[0], p1[1], w, h, label_color, 2)
            DrawObjRectFull_cairo(context, p1[0], p1[1]-text_fontsize, w, text_fontsize, label_color, 2)
            DrawObjLabelText_cairo(context, p1[0], p1[1], [255,255,255], label, text_fontsize)

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if self.pil:
            # convert to numpy first
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255).byte().cpu().numpy()
        self.im[:] = im_mask if retina_masks else scale_image(im_gpu.shape, im_mask, self.im.shape)
        if self.pil:
            # convert im back to PIL and update draw
            self.fromarray(self.im)

    def kpts(self, context, kpts, shape=(640, 640), radius=5, kpt_line=True):
        if self.pil:
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose  
        """
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
        """
        context.set_line_width(3)
        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                limb_color = [int(x) for x in self.limb_color[i]]
                context.set_source_rgb(limb_color[2],limb_color[1],limb_color[0])
                context.move_to(pos1[0], pos1[1]); context.line_to(pos2[0], pos2[1]);  context.stroke()

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image (PIL-only)
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

def check_pil_font(font=FONT, size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:  # download if missing
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')  # known issue https://github.com/ultralytics/yolov5/issues/5374
        except URLError:  # not online
            return ImageFont.load_default()

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def crop_mask(masks, boxes): # For Segmeatation 
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, shape, upsample=False): # For Segmeatation 
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)
    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = torch.nn.functional.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

def clip_coords(coords, shape): # YOLOv8 Add
    if isinstance(coords, torch.Tensor):  # faster individually
        coords[..., 0].clamp_(0, shape[1])  # x
        coords[..., 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True): # YOLOv8 Add
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords

def scale_image(im1_shape, masks, im0_shape, ratio_pad=None): # For Segmeatation 
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

#-------------------------------------------------------------------------------------------------------------------
# ADAS(Lane Detect)
#-------------------------------------------------------------------------------------------------------------------

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]

class ModelConfig():
	def __init__(self):
		self.img_w = 1472
		self.img_h = 828
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

def draw_lanes(input_img, lane_points_mat, lanes_detected, cfg, draw_points=True):
  # Write the detected line points in the image
  visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

  # Draw a mask for the current lane
  if(lanes_detected[1] and lanes_detected[2]):
    lane_segment_img = visualization_img.copy()
    cv2.fillPoly(lane_segment_img, pts = [np.vstack((lane_points_mat[1],np.flipud(lane_points_mat[2])))], color =(255,191,0))
    visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
  
  if(draw_points):
    for lane_num,lane_points in enumerate(lane_points_mat):
      for lane_point in lane_points:
        cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

  return visualization_img