from pix_2_pix.pipeline_pix2pix import PipelinePix2Pix
from pix_2_pix import constants as cn
from pix_2_pix.data import prepare_dataset_train, prepare_dataset_test
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from pix_2_pix.utils import generate_images

#path_to_data_folder = "/media/user/D/CourseWorkData"


def train(path_to_data_folder, train_csv_file, test_csv_file):
    tf.executing_eagerly()
    train_ds = prepare_dataset_train(train_csv_file, path_to_data_folder)
    test_ds = prepare_dataset_test(test_csv_file, path_to_data_folder)

    pipeline = PipelinePix2Pix()
    pipeline.fit(train_ds, cn.EPOCHS, test_ds)


def test(path_to_data_folder, test_csv_file):
    rgb = []
    depth = []
    test_ds = prepare_dataset_test(test_csv_file, path_to_data_folder)
    for d in test_ds:
        rgb.append(d[0][0].numpy())
        depth.append(d[1][0].numpy())

    rgb = np.array(rgb)
    depth = np.array(depth)

    pipeline = PipelinePix2Pix()
    path = '/media/user/D/CourseWorkData/results/ckpts_pix2pix/ckpt-18'
    pipeline.restore(path)

    pipeline.evaluate(rgb, depth)


def eval(path_to_data_folder, test_csv_file):
    rgb = []
    depth = []
    test_ds = prepare_dataset_test(test_csv_file, path_to_data_folder)
    for d in test_ds:
        rgb.append(d[0][0].numpy())
        depth.append(d[1][0].numpy())

    rgb = np.array(rgb)
    depth = np.array(depth)

    pipeline = PipelinePix2Pix()
    tf.keras.utils.plot_model(pipeline.generator, show_shapes=True, show_layer_names=False, dpi=128)
    # path = '/media/user/D/CourseWorkData/results/ckpts_pix2pix/ckpt-21'
    # pipeline.restore(path)
    # for i in range(len(rgb)):
    #         generate_images(pipeline.generator, np.reshape(rgb[i], (1, rgb[i].shape[0], rgb[i].shape[1], rgb[i].shape[2])),
    #                     np.reshape(depth[i], (1, depth[i].shape[0], depth[i].shape[1])), epoch=i, train=True)



path_to_data_folder = "/media/user/D/CourseWorkData/depth_estimation_data"
train_csv_file = os.path.join(path_to_data_folder, "data/nyu2_train.csv")
test_csv_file = os.path.join(path_to_data_folder, "data/nyu_generated_test.csv")
    # path_rgb = os.path.join(path_to_data_folder, 'eigen_test_rgb.npy')
    # path_depth = os.path.join(path_to_data_folder, 'eigen_test_depth.npy')

if os.path.exists(train_csv_file) and os.path.exists(test_csv_file):
  #  train(path_to_data_folder, train_csv_file, test_csv_file)
    eval(path_to_data_folder, test_csv_file)
else:
    print("Input .csv files don't exist")



