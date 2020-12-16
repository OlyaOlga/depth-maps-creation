from dense_depth.data import DataLoader
import tensorflow as tf
from dense_depth.loss import depth_loss_function
import os
import sys
from datetime import  datetime
from dense_depth.model import DepthEstimate
from dense_depth.evaluate import load_test_data, evaluate
import matplotlib.pyplot as plt
import numpy as np

batch_size     = 2
learning_rate  = 0.0001
epochs         = 20



def train(path_to_data_folder):
    model = DepthEstimate()
    dl = DataLoader(path_to_data_folder=path_to_data_folder, DEBUG=True)
    train_generator = dl.get_batched_dataset(batch_size)

    print('Data loader ready.')

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)

    model.compile(loss=depth_loss_function, optimizer=optimizer)



    checkpoint_path = "output/ckpts/training_{epoch}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=10)

    logdir = "output/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(train_generator, epochs=epochs, steps_per_epoch=dl.length//batch_size, callbacks=[cp_callback, tensorboard_callback])


def test(path_to_eval_data):
    model = DepthEstimate()
    checkpoint_path = f"training_{epochs}/cp.ckpt"
    model.load_weights(checkpoint_path)
    print('Model weights loaded.')
    rgb, depth, crop = load_test_data(path_to_eval_data)

    evaluate(model, rgb, depth, crop)


def save_img(test_input, tar, prediction, num):
    plt.figure(figsize=(15, 15))

    display_list = [test_input, tar, prediction]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.savefig(f'output/{num}.png')


def eval(path_to_eval_data):
    model = DepthEstimate()
    checkpoint_path = '/media/user/D/CourseWorkData/results/dense_depth/training_15/cp.ckpt'
    model.load_weights(checkpoint_path).expect_partial()

    rgb, depth, crop = load_test_data(path_to_eval_data)
    rgb = rgb[100:112, :, :, :]
    depth = depth[100:112, :, :]

    output = evaluate(model, rgb, depth, crop)
    for i in range(len(rgb)):
        np.save(f'point_clouds_generation/real_{i}.npy', depth[i]/10)
        np.save(f'point_clouds_generation/generated_{i}.npy', output[i]/10)
        np.save(f'point_clouds_generation/rgb_{i}.npy', rgb[i])

        print(i)



def main():
    path_to_data_folder = sys.argv[1]#'/media/user/D/CourseWorkData/depth_estimation_data'
    path_to_eval_data = os.path.join(path_to_data_folder, 'nyu_test.zip')
    #train(path_to_data_folder)
    eval(sys.argv[1])


if __name__ == '__main__':
    main()
