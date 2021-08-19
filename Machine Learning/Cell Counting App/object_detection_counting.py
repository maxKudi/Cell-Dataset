######## Cell Detection & Counter Using Tensorflow-trained Classifier #########
#
# Author: Chukwudi Okerulu
# Date: 10/07/21
# Description: 
# This program uses a TensorFlow-trained neural network to perform cell detection and further code to implement cell counting.
# It loads the classifier and uses it to perform cell detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.
## Notable websites:
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


# Import packages
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import pathlib
import glob
import re
import math
import fnmatch
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()


####################################################################################################################################################################################################################################################################################################################################################################################################
# Sorting images into a new folder and change filename
ORIGINAL_DIR = os.path.join(CWD_PATH, 'test_images' , 'original_images')
#print(len([name for name in os.listdir(ORIGINAL_DIR) if os.path.isfile(os.path.join(ORIGINAL_DIR, name))]))
#TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


# Myimage_files = []

# if os.path.isdir(ORIGINAL_DIR):
#         for root, dirs, files in os.walk(ORIGINAL_DIR):
#             for IMAGE_NAME in fnmatch.filter(files, '*.png'):
#                 Imagefpath = os.path.join(root, IMAGE_NAME)
#                 Myimage_files.append(Imagefpath)
#                 for i in range(0,len(Myimage_files)):
#                     img = Image.open(Myimage_files[i])
#                     new_path_name = os.path.join(CWD_PATH,'test_images', 'A' + str(i) + ".png")
#                     img.save(new_path_name)


      
for alpbt in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':

    if(pathlib.Path(os.path.join(CWD_PATH,'test_images', 'original_images',alpbt)))!= None :
        for i in range(1,5):
            PATH_TO_TEST_IMAGES_DIR = pathlib.Path(os.path.join(CWD_PATH,'test_images','original_images',alpbt + str(i)))
            TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.png")))
            #IMAGE_NAME = os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.png")
            for IMAGE_NAME in TEST_IMAGE_PATHS:
                img = Image.open(IMAGE_NAME)
                complete_name = os.path.join(CWD_PATH,'test_images', alpbt + str(i) + ".png")
                img.save(complete_name)

    else:
        break


####################################################################################################################################################################################################################################################################################################################################################################################################
# Cropping images into smaller images for better detection
# Crops the image into 16 grid and saves it as "new_filename" in 220*220 pixel 
def crop_image(img, crop_area, new_filename):
    cropped_image = img.crop(crop_area)
    cropped_image.save(new_filename)
# The x, y coordinates of the areas to be cropped. (x1, y1, x2, y2)(left,upper,right,low)

lower_area_1_1=1954
lower_area_1_2=1734
upper_area_1_1= 1734
upper_area_1_2 = 1514
crop_areas_1 = [(100, upper_area_1_1, 320, lower_area_1_1), (520, upper_area_1_1, 740, lower_area_1_1), (740, upper_area_1_1, 960, lower_area_1_1), (960, upper_area_1_1, 1180, lower_area_1_1), (1180, upper_area_1_1, 1400, lower_area_1_1), (1400, upper_area_1_1, 1620, lower_area_1_1), (1620, upper_area_1_1, 1840, lower_area_1_1), (1840, upper_area_1_1, 2060, lower_area_1_1),
                (100, upper_area_1_2, 320, lower_area_1_2), (520, upper_area_1_2, 740, lower_area_1_2), (740, upper_area_1_2, 960, lower_area_1_2), (960, upper_area_1_2, 1180, lower_area_1_2), (1180, upper_area_1_2, 1400, lower_area_1_2), (1400, upper_area_1_2, 1620, lower_area_1_2), (1620, upper_area_1_2, 1840, lower_area_1_2), (1840, upper_area_1_2, 2060, lower_area_1_2)]

crop_areas_2 = [(300, upper_area_1_1, 520, lower_area_1_1), (520, upper_area_1_1, 740, lower_area_1_1), (740, upper_area_1_1, 960, lower_area_1_1), (960, upper_area_1_1, 1180, lower_area_1_1), (1180, upper_area_1_1, 1400, lower_area_1_1), (1400, upper_area_1_1, 1620, lower_area_1_1), (1620, upper_area_1_1, 1840, lower_area_1_1), (1840, upper_area_1_1, 2060, lower_area_1_1),
                (300, upper_area_1_2, 520, lower_area_1_2), (520, upper_area_1_2, 740, lower_area_1_2), (740, upper_area_1_2, 960, lower_area_1_2), (960, upper_area_1_2, 1180, lower_area_1_2), (1180, upper_area_1_2, 1400, lower_area_1_2), (1400, upper_area_1_2, 1620, lower_area_1_2), (1620, upper_area_1_2, 1840, lower_area_1_2), (1840, upper_area_1_2, 2060, lower_area_1_2)]

left_area_1_1=1900
left_area_1_2=2120
right_area_1_1= 2120
right_area_1_2 = 2340
crop_areas_4 = [(left_area_1_1, 100, right_area_1_1, 320), (left_area_1_1, 320, right_area_1_1, 540), (left_area_1_1, 540, right_area_1_1, 760), (left_area_1_1, 760, right_area_1_1, 980), (left_area_1_1, 980, right_area_1_1, 1200), (left_area_1_1, 1200, right_area_1_1, 1420), (left_area_1_1, 1420, right_area_1_1, 1640), (left_area_1_1, 1640, right_area_1_1, 1860),
                (left_area_1_2, 100, right_area_1_2, 320), (left_area_1_2, 320, right_area_1_2, 540), (left_area_1_2, 540, right_area_1_2, 760), (left_area_1_2, 760, right_area_1_2, 980), (left_area_1_2, 980, right_area_1_2, 1200), (left_area_1_2, 1200, right_area_1_2, 1420), (left_area_1_2, 1420, right_area_1_2, 1640), (left_area_1_2, 1640, right_area_1_2, 1860)]

left_area_3_1=0
left_area_3_2=220
right_area_3_1= 220
right_area_3_2 = 440
crop_areas_3 = [(left_area_3_1, 100, right_area_3_1, 320), (left_area_3_1, 320, right_area_3_1, 540), (left_area_3_1, 540, right_area_3_1, 760), (left_area_3_1, 760, right_area_3_1, 980), (left_area_3_1, 980, right_area_3_1, 1200), (left_area_3_1, 1200, right_area_3_1, 1420), (left_area_3_1, 1420, right_area_3_1, 1640), (left_area_3_1, 1640, right_area_3_1, 1860),
                (left_area_3_2, 100, right_area_3_2, 320), (left_area_3_2, 320, right_area_3_2, 540), (left_area_3_2, 540, right_area_3_2, 760), (left_area_3_2, 760, right_area_3_2, 980), (left_area_3_2, 980, right_area_3_2, 1200), (left_area_3_2, 1200, right_area_3_2, 1420), (left_area_3_2, 1420, right_area_3_2, 1640), (left_area_3_2, 1640, right_area_3_2, 1860)]



# Find all png images in a directory
imgList=glob.glob(os.path.join(CWD_PATH,'test_images','*1.png'))         
alpbt = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"    #the alphabet for the image labeling
save_path = os.path.join(CWD_PATH,'test_images','cropped_images')
if not os.path.exists(save_path):
    os.makedirs(save_path)

pic_count=0
for img in imgList:                             #Loop through all found images
    image = Image.open(img)                     #open the image
    pic_name = alpbt[pic_count]+ "1_crop"       #for the label of the image correspond to each alphabet
    pic_count +=1

    # Loops through the "crop_areas" list and crops the image based on the coordinates in the list 
    for i, crop_area in enumerate(crop_areas_1):
        filename = os.path.splitext(img)[0]
        ext = os.path.splitext(img)[1]
        #new_filename = filename + '_' + str(i) + ext
        new_filename = os.path.join(save_path,pic_name + "_" + str(i) + ".png")
        cropped_image = image.crop(crop_area)
        cropped_image.save(new_filename)
        #crop_image(image, crop_area, new_filename)


# Find all png images in a directory
imgList=glob.glob(os.path.join(CWD_PATH,'test_images','*2.png'))         
pic_count=0
for img in imgList:                         #Loop through all found images
    image = Image.open(img)                 #open the image
    pic_name = alpbt[pic_count]+ "2_crop"   #for the label of the image correspond to each alphabet
    pic_count +=1
    
    # Loops through the "crop_areas" list and crops the image based on the coordinates in the list
    for i, crop_area in enumerate(crop_areas_2):
        filename = os.path.splitext(img)[0]
        ext = os.path.splitext(img)[1]
        #new_filename = filename + '_' + str(i) + ext
        new_filename = os.path.join(save_path,pic_name + "_" + str(i) + ".png")
        cropped_image = image.crop(crop_area)
        cropped_image.save(new_filename)
        #crop_image(image, crop_area, new_filename)

# Find all png images in a directory
imgList=glob.glob(os.path.join(CWD_PATH,'test_images','*3.png'))         
pic_count=0
for img in imgList:                         #Loop through all found images
    image = Image.open(img)                 #open the image
    pic_name = alpbt[pic_count]+ "3_crop"   #for the label of the image correspond to each alphabet
    pic_count +=1
    
    # Loops through the "crop_areas" list and crops the image based on the coordinates in the list
    for i, crop_area in enumerate(crop_areas_3):
        filename = os.path.splitext(img)[0]
        ext = os.path.splitext(img)[1]
        #new_filename = filename + '_' + str(i) + ext
        new_filename = os.path.join(save_path,pic_name + "_" + str(i) + ".png")
        cropped_image = image.crop(crop_area)
        cropped_image.save(new_filename)
        #crop_image(image, crop_area, new_filename)


#Find all png images in a directory
imgList=glob.glob(os.path.join(CWD_PATH,'test_images','*4.png'))         
pic_count=0
for img in imgList:                         #Loop through all found images
    image = Image.open(img)                 #open the image
    pic_name = alpbt[pic_count]+ "4_crop"   #for the label of the image correspond to each alphabet
    pic_count +=1
     
# Loops through the "crop_areas" list and crops the image based on the coordinates in the list
    for i, crop_area in enumerate(crop_areas_4):
        filename = os.path.splitext(img)[0]
        ext = os.path.splitext(img)[1]
        #new_filename = filename + '_' + str(i) + ext
        new_filename = os.path.join(save_path,pic_name + "_" + str(i) + ".png")
        cropped_image = image.crop(crop_area)
        cropped_image.save(new_filename)
        #crop_image(image, crop_area, new_filename)




##################################################################################################################################################################################################################################################################################################
# Code for Object detection and counting 
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'image1_0.png'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
# PATH_TO_IMAGE = os.path.join(CWD_PATH,'test_images', IMAGE_NAME)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path(os.path.join(CWD_PATH,'cropped_images'))
#Make an array that stores all images
#TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob('*.png')))


# Number of classes the object detector can identify
NUM_CLASSES = 2
k = 0

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `1`, we know that this corresponds to `white`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)



imgList=glob.glob(os.path.join(CWD_PATH ,'test_images','cropped_images','*.png'))
for PATH_TO_IMAGE in imgList:
    # Define input and output tensors (i.e. data) for the object detection classifier
    #print(PATH_TO_IMAGE)
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.60)

    #save to same folder as data input
    Output_image_path = os.path.join(CWD_PATH ,'test_images','output_images')
    k += 1
    if not os.path.exists(Output_image_path):
        os.makedirs(Output_image_path)
    
    Output_image_path = os.path.join(CWD_PATH ,'test_images','output_images', ' ')
    img = Image.fromarray(image)
    img.save(Output_image_path + str(k) +'.png')
        

    # All the results have been drawn on image. Now display the image.
    #cv2.imshow('Object detector', image)

    # Press any key to close the image
    #cv2.waitKey(0)

    # Clean up
    #cv2.destroyAllWindows()



####################################################################################################################################################################################################################################################################################################################################################################################################
# Code to combine the result of white and black cell count and output final result txt.file
# Accessing the result.txt file
PATH_TO_RESULT = os.path.join(CWD_PATH, 'result.txt')
PATH_TO_FINAL_RESULT = os.path.join(CWD_PATH, 'final_result.txt')
result_file = open(PATH_TO_RESULT,"r")

# Initialization of final result file
final_result_file = open(PATH_TO_FINAL_RESULT, "w")
final_result_file.close()

# Reading each line on the result.txt and put it into an array-list
lines = result_file.readlines()
# Initialization of the amount of cells
white_amount = 0
black_amount = 0

# Opening the final result file to add data
final_result_file = open(PATH_TO_FINAL_RESULT, "a")


# Iterating through each line of result.txt
save_path = os.path.join(CWD_PATH,'test_images','cropped_images')
for i in range(0,len([name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name))])): # The range need to be change according to desired amount of cropped images per complete image
    #print(lines[i])
    single_line = lines[i]
    #schwarz_line = lines[i]
    #word, number = single_line.split()
    #print(number)

    weiss = re.search('\'L_Cell\': (\d+)', single_line, re.IGNORECASE)
    schwarz = re.search('\'D_Cell\': (\d+)', single_line, re.IGNORECASE)
    #leer = re.search('{}', single_line)
    if single_line.find('L_Cell') >= 0 and single_line.find('D_Cell') >= 0 :
        #print(weiss.group(1))
        white_amount += int(weiss.group(1))
        #print(schwarz.group(1))
        black_amount += int(schwarz.group(1))
    elif single_line.find('L_Cell') >= 0:
        #print(weiss.group(1))
        white_amount += int(weiss.group(1))
    elif single_line.find('D_Cell') >= 0:
        #print(schwarz.group(1))
        black_amount += int(schwarz.group(1))
    else:
        black_amount += 0
        white_amount += 0
    
# Writing the final result into the .txt data file
total_amount = white_amount + black_amount
white_per = (white_amount/total_amount)*100
black_per = (black_amount/total_amount)*100

final_result_file.write("Amount of Living cells : " + str(white_amount) +
                        "\n" + "Amount of Dead cells : " + str(black_amount) +
                        "\n\n" + "Average of Living cells : " + str(white_amount/4) +
                        "\n" + "Average of Dead cells : " + str(black_amount/4) +
                        "\n\n" + "Percentage of Living cells : " + str(round(white_per)) + "%" +
                        "\n" + "Percentage of Dead cells : " + str(round(black_per)) + "%")


final_result_file.close()

result_file.close()


