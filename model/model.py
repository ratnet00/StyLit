#dependencies
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2

#model from hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

#preprocess image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

#load required images
content_image = load_image('/home/ritvikt/stylit/images/profile.jpg')
style_image = load_image('/home/ritvikt/stylit/images/monet.jpeg')

#generate result
stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

#save image
cv2.imwrite('generated_img.jpg', cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

