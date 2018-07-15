
import tensorflow as tf
import PIL
import numpy as np
import json
import base64

flags = tf.app.flags
flags.DEFINE_string('input_image', '', 'Path to an input image')
flags.DEFINE_string('output_file', '', 'Path to the output image')
FLAGS = flags.FLAGS

# this method doesn't work and will result in "Request payload size exceeds the limit:"
def export_image_as_array(input_image, output_file):
    image = PIL.Image.open(input_image)
    image_expanded = np.expand_dims(image, axis=0)
    image_expanded_list = image_expanded.tolist()
    input_json = {
        "inputs": image_expanded_list
    }
    with file(output_file, "w") as outfile:
        json.dump(input_json, outfile)

# this method won't work because I can't find an easy way to export arbitrary
# tensors to a JSON file
def export_image_tensorflow(input_image, output_file):
    image_decoded = tf.image.decode_jpeg(tf.read_file(input_image), channels=3)
    enc = tf.image.encode_png(image_decoded)
    fname = tf.constant(output_file)
    fwrite = tf.write_file(fname, enc)

    sess = tf.Session()
    result = sess.run(fwrite)

def export_image(input_image, output_file):
    with open(input_image, 'rb') as infile:
        contents = infile.read()
        b64contents = base64.urlsafe_b64encode(contents)
    # input_json = b64contents
    input_json = {
        "inputs": b64contents
    }
    with file(output_file, 'w') as outfile:
        json.dump(input_json, outfile)

def main(_):
    export_image(FLAGS.input_image, FLAGS.output_file)
    
if __name__ == '__main__':
    tf.app.run()
