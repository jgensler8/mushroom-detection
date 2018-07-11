######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
# import cv2
import PIL
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
import label_map_util
import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('label_map_file', '', 'Path to the label map .pbtxt')
flags.DEFINE_string('input_image', '', 'Path to an input image')
flags.DEFINE_string('output_image', '', 'Path to the output image')
flags.DEFINE_float('score_threshold', '.2', 'Threshold that classes need to be labeled')
flags.DEFINE_string('inference_graph_dir', 'inference_graph', 'directory conatining an inference graph')
FLAGS = flags.FLAGS

def detect(label_map_file, input_image, output_image, inference_graph_dir, score_threshold, num_classes=1, line_thickness=4):
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(inference_graph_dir, 'frozen_inference_graph.pb')

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(label_map_file)
    print(label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
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

    # Define input and output tensors (i.e. data) for the object detection classifier

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
    # image = cv2.imread(input_image)
    image = PIL.Image.open(input_image)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    print("~~~Boxes~~~")
    print(np.squeeze(boxes))
    print("~~~Scores~~~")
    print(np.squeeze(scores))
    print("~~~Classes~~~")
    print(np.squeeze(classes))
    print("~~~Num~~~")
    print(num)
    print("~~~Score Theshold~~~")
    print(score_threshold)
    
    final_boxes = []
    
    scores = np.squeeze(scores)
    boxes = np.squeeze(boxes)
    
    for index in range(0, num):
        if scores[index] > score_threshold:
            final_boxes.append(boxes[index])

    # vis_util.visualize_boxes_and_labels_on_image(
    #     image,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=line_thickness,
    #     min_score_thresh=score_threshold)
    
    vis_util.draw_bounding_boxes_on_image(
        image,
        np.array(final_boxes))
        
    image.save(output_image, "JPEG")
    # output = PIL.Image.open(output_image)
    # output.save
    # cv2.imwrite(output_image, image)

    # All the results have been drawn on image. Now display the image.
    # cv2.imshow('Object detector', image)

    # # Press any key to close the image
    # cv2.waitKey(0)

    # Clean up
    # cv2.destroyAllWindows()
    
    
def main(_):
    detect(FLAGS.label_map_file, FLAGS.input_image, FLAGS.output_image, FLAGS.inference_graph_dir, FLAGS.score_threshold)
    
if __name__ == '__main__':
    tf.app.run()
