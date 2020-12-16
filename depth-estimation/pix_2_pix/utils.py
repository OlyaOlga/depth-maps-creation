import matplotlib.pyplot as plt
import pix_2_pix.constants as cn
import numpy as np
from skimage.transform import resize
from time import time

def generate_images(model, test_input, tar, epoch=None, train=True):
  begin = time()
  prediction = model(test_input, training=False)
  end = time()

  print(end-begin)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  if train is True:
    plt.savefig(cn.LOG_DIR+str(epoch)+'.png')
  else:
    plt.show()

def convert_output_to_meters(depth):
  depth_in_meters = (depth+1)*5
  return depth_in_meters


def convert_input_to_meters(depth):
  depth_in_meters = depth*(10/255.0)
  return depth_in_meters

def scale_up(output_shape, images):
    scaled = []
    for i in range(len(images)):
      img = images[i]
      scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)