import os

import tensorflow as tf
from sklearn.utils import shuffle

from . import constants as cn

def load(image_file, image_label):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  color_image = tf.cast(image, tf.float32)

  label = tf.io.read_file(image_label)
  label = tf.io.decode_png(label)
  label = tf.cast(label, tf.float32)

  return color_image, label


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  print('resizing')

  return input_image, real_image


def random_crop(input_image, real_image):
#  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_input_image = tf.image.random_crop(
      input_image, size=[cn.IMG_HEIGHT, cn.IMG_WIDTH, 3])

  cropped_real_image = tf.image.random_crop(
      real_image, size=[cn.IMG_HEIGHT, cn.IMG_WIDTH, 1])

  return cropped_input_image, cropped_real_image

# normalizing the images to [-1, 1]
@tf.function()
def normalize(input_image, real_image):
  input_image =  (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(image_file, image_label):
  input_image, real_image = load(image_file, image_label)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file, image_label):
  input_image, real_image = load(image_file, image_label)
  input_image, real_image = resize(input_image, real_image, cn.IMG_HEIGHT, cn.IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def read_nyu_data(csv_file, path_to_data_folder, DEBUG=cn.IS_DEBUG):
    csv = open(csv_file, 'r').read()
    nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

    # Dataset shuffling happens here
    nyu2_train = shuffle(nyu2_train, random_state=0)

    # Test on a smaller dataset
    if DEBUG: nyu2_train = nyu2_train[:4]

    # A vector of RGB filenames.
    filenames = [os.path.join(path_to_data_folder, i[0]) for i in nyu2_train]

    # A vector of depth filenames.
    labels = [os.path.join(path_to_data_folder, i[1]) for i in nyu2_train]

    return filenames, labels

def prepare_dataset_train(train_csv_file, path_to_data_folder):
    filenames, labels = read_nyu_data(train_csv_file, path_to_data_folder, DEBUG=cn.IS_DEBUG)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(cn.BUFFER_SIZE)
    dataset = dataset.batch(cn.BATCH_SIZE)

    return dataset


def prepare_dataset_test(train_csv_file, path_to_data_folder):
    filenames, labels = read_nyu_data(train_csv_file, path_to_data_folder, DEBUG=cn.IS_DEBUG)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(cn.BATCH_SIZE)

    return dataset
