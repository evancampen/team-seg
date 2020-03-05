import os
import sys
import time
import re
import skimage
import imageio
sys.path.append(".")
import numpy as np
import tensorflow as tf

# Import Mask RCNN
from config import Config
import utils
import model as modellib
import visualize
import coco
from skimage import exposure
import matplotlib.pyplot as plt
from PIL import Image


def imageLoader(batch_size, path, file_name,i):
    img =[]
    for n in range(batch_size):
        img.append(skimage.io.imread(path+file_name[i+n],plugin='imageio'))
    return img
    
def batchVisualizer(batch_size, image_list, file_name, 
                    i,results, save_path,class_names):
    if (i>len(file_name)): 
        return
    for n in range(batch_size):
        if not os.path.exists(save_path+file_name[i+n]+"/"):
            os.mkdir(save_path+file_name[i+n]+"/")
        visualize.save_masked_image(image_list[n], file_name[i+n] ,
                                    results[n]['rois'], results[n]['masks'],
                                    results[n]['class_ids'], results[n]['scores'],
                                    class_names,
                                    filter_classs_names='person',
                                    save_dir =save_path+file_name[i+n]+"/")
    
def batchDir(batch_size, path,file_name): #create directory for segmentation
    save_path_temp = path + file_name[i]+"/"
    if not os.path.exists(save_path_temp):
        os.mkdir(save_path_temp)


# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"
# Download COCO trained weights from Releases
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
if not os.path.exists("logs"):
    os.mkdir("logs")
    
class inferenceConfig(coco.CocoConfig):
    #Single image processing
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

#create model in inference mode
model_rcnn = modellib.MaskRCNN(mode = "inference", model_dir="logs/", config = inferenceConfig())
print('Loading MaskRCNN model')
print('########################################################')
model_rcnn.load_weights(COCO_MODEL_PATH, by_name = True)
# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#list of input folder paths
folder_path_image = sys.argv[1]

#save path for segmented images
save_path_seg = sys.argv[2]

#save path for txt xyz and ply pointcloud
file_name = []
if len(sys.argv) is 3:
    file_names = os.listdir(folder_path_image)
    for file in file_names:
        if file.endswith((".png",".bmp",".jpg")):
            file_name.append(file)
else:
    file_name.append(sys.argv[3]) #file names of images to process
file_name.sort()

times = []
batch_no=4
for i in range(0,len(file_name),batch_no):
    print("starting file {}\n".format(file_name[i]))
    if (i>36): 
        continue
    batchDir(batch_no,save_path_seg, file_name)
    
    #stacks images for gpu input
    image_list =imageLoader(batch_no, folder_path_image, file_name,i)
    image_list = np.asarray(image_list)
        
    print("#########################################################")
    start = time.time()
    results = model_rcnn.detect(image_list, verbose=0)
    end = time.time()
    
    print("time elapsed per img:" + str((end-start)/batch_no))
    times.append(end-start)
    
    #saves segmented images and mask
    start = time.time()
    batchVisualizer(batch_no,image_list, file_name, 
                    i,results, save_path_seg,class_names)  
    end = time.time()
    
    print("processing time elapsed per img:" + str((end-start)/batch_no))    
print("average time per process: " +str(sum(times)/len(times)/batch_no))
