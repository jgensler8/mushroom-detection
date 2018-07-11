"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import glob
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from PIL import Image
from collections import namedtuple, OrderedDict
import dataset_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'mushroom':
        return 1
    else:
        None

def create_tf_example(image_file):
    print("opening {}".format(image_file))
    with tf.gfile.GFile(image_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = os.path.basename(image_file).encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
        
    xml_file = image_file + ".xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        classes_text.append(member[0].text.encode('utf-8'))
        classes.append(class_text_to_int(member[0].text))
        xmins.append(int(member[5][0].text) / width)
        xmaxs.append(int(member[5][1].text) / width)
        ymins.append(int(member[5][2].text) / height)
        ymaxs.append(int(member[5][3].text) / height)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for image in glob.glob(os.path.join(FLAGS.image_dir, '*.jpeg')):
        tf_example = create_tf_example(image)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()