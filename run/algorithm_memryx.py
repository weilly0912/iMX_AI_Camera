import numpy as np
import os
import csv
import cv2
import cairo
import threading
import tflite_runtime.interpreter as tflite #[RK3588] Remove
from utils import *
from memryx import NeuralCompiler, AsyncAccl, SyncAccl, MultiStreamAsyncAccl

# --------------------------------------------------------------------------------------------------------
# Model Arch
# --------------------------------------------------------------------------------------------------------

coco_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class YoloV8:
    """
    A helper class to run YOLOv8 pre- and post-proccessing.
    """
    def __init__(self, stream_img_size=None):
        """
        The initialization function.
        """

        self.name = 'YoloV8'
        self.input_size = (640,640,3) 
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.4
        self.iou_thres = 0.6

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True
    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.img = img

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))
        
        img = img.astype(np.float32)
        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0


        # Expand dimensions to add the batch size as the third axis
        image_data = np.expand_dims(image_data, axis=2)  # Adds batch dimension after width and height
        image_data = np.expand_dims(image_data, axis=0)  # Adds another dimension for batch size

        # Return the preprocessed image data
        return image_data
    def postprocess(self, output):
        """
        Performs post-processing on the YOLOv8 model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            list: A list of detections where each detection is a dictionary containing 
                    'bbox', 'class_id', 'class', and 'score'.
        """
        # Transpose the output to shape (8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Extract the bounding box information and class scores in a vectorized manner
        boxes = outputs[:, :4]  # (8400, 4) - x_center, y_center, width, height
        class_scores = outputs[:, 4:]  # (8400, 80) - class scores for 80 classes

        # Find the class with the highest score for each detection
        max_scores = np.max(class_scores, axis=1)  # (8400,) - maximum class score for each detection
        class_ids = np.argmax(class_scores, axis=1)  # (8400,) - index of the best class

        # Filter out detections with scores below the confidence threshold
        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []  # Return an empty list if no valid detections

        # Select only valid detections
        valid_boxes = boxes[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]

        # Convert bounding box coordinates from (x_center, y_center, w, h) to (left, top, width, height)
        valid_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor  # left
        valid_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor  # top
        valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor  # width
        valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor  # height

        # Create detection dictionaries
        detections = [{
            'bbox': valid_boxes[i].astype(int).tolist(),
            'class_id': int(valid_class_ids[i]),
            'class': coco_dict[int(valid_class_ids[i])],
            'score': valid_scores[i]
        } for i in range(len(valid_indices))]

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if len(detections) > 0:
            # NMS requires two lists: bounding boxes and confidence scores
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres)

            # Check if indices is not empty
            if len(indices) > 0:
                # Flatten indices if they are returned as a list of arrays
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]

                # Filter detections based on NMS
                final_detections = [detections[i] for i in indices]
            else:
                final_detections = []
        else:
            final_detections = []

        # Return the list of final detections
        return final_detections

# --------------------------------------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------------------------------------

def mx_create_objectdetect_YOLOv8n( dfp_path, post_path ):
    model = YoloV8()
    accl = SyncAccl(dfp=dfp_path)
    accl_post = {}
    accl_post['interpreter'] = tflite.Interpreter(model_path=post_path)
    accl_post['interpreter'].allocate_tensors()
    accl_post['input_details'] = accl_post['interpreter'].get_input_details()
    accl_post['output_details'] = accl_post['interpreter'].get_output_details()
    return model, accl, accl_post

# --------------------------------------------------------------------------------------------------------
# DEMO 1.0 : MemryX - Objection detection (YOLOv8n)
# * Code Ver : 1.0
# * Code Date: 2025/03/19
# * Author   : Weilly Li
# --------------------------------------------------------------------------------------------------------

def mx_detect_objectdetect_YOLOv8n(accl, accl_post, model, frame, PicturePath, IsLoadPicture):

    FrameInput = cv2.imread(PicturePath) if IsLoadPicture else frame
    input_data = model.preprocess(FrameInput)
    outputs = accl.run(input_data, model_idx=0)
    
    accl_post['interpreter'].set_tensor(accl_post['input_details'][0]['index'], np.expand_dims(outputs[0], axis=0))
    accl_post['interpreter'].set_tensor(accl_post['input_details'][1]['index'], np.expand_dims(outputs[1], axis=0))
    accl_post['interpreter'].set_tensor(accl_post['input_details'][2]['index'], np.expand_dims(outputs[2], axis=0))
    accl_post['interpreter'].set_tensor(accl_post['input_details'][3]['index'], np.expand_dims(outputs[3], axis=0))
    accl_post['interpreter'].set_tensor(accl_post['input_details'][4]['index'], np.expand_dims(outputs[4], axis=0))
    accl_post['interpreter'].set_tensor(accl_post['input_details'][5]['index'], np.expand_dims(outputs[5], axis=0))
    accl_post['interpreter'].invoke()
    postprocessed_output = accl_post['interpreter'].get_tensor(accl_post['output_details'][0]['index'])
    detections = model.postprocess(postprocessed_output)

    # Output Setting
    width = FrameInput.shape[1]
    height = FrameInput.shape[0]
    InfereneTime = 1
    pred = detections

    # Destination of Buffer
    FrameOutput = FrameInput #GetDstBuffer(FrameInput, width, height, IsLoadPicture)

    return [FrameOutput, [[width, height], pred], InfereneTime]

def mx_draw_objectdetect_YOLOv8n(context, frame, drawout_details):
	# DrawOut
	pred  = drawout_details[1]
	labels_coco   = load_labels("../data/label/coco_labels.txt") 
	colors_coco   = generate_colors(labels_coco)
	rect_linewidth      = 4
	label_text_fontsize = 24
	for detection in pred:
		box = detection["bbox"]  
		score = detection["score"]  
		class_id = detection["class_id"]
		x, y, w, h = box
		color = colors_coco[int(class_id) % len(colors_coco)]; 
		label = coco_dict[class_id]
		line_width=rect_linewidth; font_size = label_text_fontsize
		DrawObjRect_cairo(context, x, y, w, h, color, line_width)
		DrawObjLabelText_cairo(context, x, y, color, label, font_size)





