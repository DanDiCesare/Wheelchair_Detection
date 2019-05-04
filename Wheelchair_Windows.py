'''
Daniel Di Cesare - 2019
Sources from: github.com/EdjeElectronics
Modified source code for ease of use and DP4 use case
'''

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
import winsound
from time import sleep

#Needed for notebook (Jupyter)
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

#Find Module & working directory
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()

# Path to frozen graph.pb (trained model) & Label Map
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
NUM_CLASSES = 2

# Label maps map -> category names
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Model -> memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


#parameters for later
x=0
y=0
IM_WIDTH = 1280
IM_HEIGHT = 720
font = cv2.FONT_HERSHEY_DUPLEX
#Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

#Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

#Score == confidence in the image
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# #detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Webcam
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

#Bounding Box
TL_inside = (int(IM_WIDTH*0.1),int(IM_HEIGHT*0.35))
BR_inside = (int(IM_WIDTH*0.45),int(IM_HEIGHT-5))
#TL_inside = (int(IM_WIDTH*.2),int(IM_HEIGHT*.75))
#BR_inside = (int(IM_WIDTH*0.65),int(IM_HEIGHT*.2))
detected_inside = False
inside_counter = 0
pause = 0
pause_counter = 0


while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    # Draw FPS

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

    cv2.rectangle(frame,TL_inside,BR_inside,(20,20,255),3)
    cv2.putText(frame,"Detection Area",(TL_inside[0]+10,TL_inside[1]-10),font,1,(20,255,255),3,cv2.LINE_AA)

    ##Ignores boxes that are too big
    if (int(classes[0][0]) == 1) and (pause == 0) and (abs(boxes[0][0][1] - boxes[0][0][3])*IM_WIDTH < 1100) and (abs(boxes[0][0][0] - boxes[0][0][2])*IM_HEIGHT < 650):
        x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
        y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)
        # Draw a circle at center of object
        cv2.circle(frame,(x,y), 5, (75,13,180), -1)

        # If object is in inside box, increment inside counter variable
        if ((x > TL_inside[0]) and (x < BR_inside[0]) and (y > TL_inside[1]) and (y < BR_inside[1])):
            inside_counter = inside_counter + 1

    if inside_counter > 100:  ##within box for two seconds
        detected_inside = True
        inside_counter = 0
        winsound.Beep(440, 250)
        pause = 1

    if pause == 1:
        if detected_inside == True: ##ADD MOTOR HERE, maybe get rid of pause counter, just sleep with motor controls
            print("FUCK YEA")
            cv2.putText(frame,'WHEELCHAIR',(int(IM_WIDTH/2),int(IM_HEIGHT/2)),font,3,(0,0,0),7,cv2.LINE_AA)
            cv2.putText(frame,'WHEELCHAIR',(int(IM_WIDTH/2),int(IM_HEIGHT/2)),font,3,(95,176,23),5,cv2.LINE_AA)

        pause_counter = pause_counter + 1
        if pause_counter > 200:
            pause = 0
            pause_counter = 0
            detected_inside = False
            detected_outside = False

    cv2.putText(frame,'Detection counter: ' + str(max(inside_counter, 1)),(10,100),font,0.5,(255,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'Pause counter: ' + str(pause_counter),(10,150),font,0.5,(255,255,0),1,cv2.LINE_AA)
    
    ##debugging purposes
    #print(scores)
    #print(category_index)

    #Display frame
    cv2.imshow('Wheelchair detector', frame)

    #System interrupt
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

