import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def prepare_test_data():
    def norm_label(label):
        return (label*25.5).astype(np.uint8)
    path_rgb = '/media/user/D/CourseWorkData/eigen_test_rgb.npy'
    path_depth = '/media/user/D/CourseWorkData/eigen_test_depth.npy'
    foldrname = 'data/nyu_generated_test'
    output_path = '/media/user/D/CourseWorkData'

    datast_rgb = np.load(path_rgb)
    dataset_depth = np.load(path_depth)

    annotations = []

    for img, label, num in zip(datast_rgb, dataset_depth, range(len((datast_rgb)))):

        img_rel_path = os.path.join(foldrname, f'{num:04d}.jpg')
        filepath_img = os.path.join(output_path, img_rel_path)

        depth_rel_path = os.path.join(foldrname, f'{num:04d}.png')
        filepath_label = os.path.join(output_path, depth_rel_path)

        label_normed = norm_label(label)
        current_sample = img_rel_path+','+depth_rel_path

        annotations.append(current_sample)
        # cv2.imwrite(filepath_img, img)
        # cv2.imwrite(filepath_label, label_normed)

    annotations = np.array(annotations)
    path = os.path.join(output_path, foldrname+'.csv')

    np.savetxt(path, annotations, delimiter='\n', fmt='%s')

path_to_csv = '/media/user/D/CourseWorkData/results/pix2pix_epochs.csv'

df = pd.read_csv(path_to_csv, sep=',     ')
df.index = df.index+1

print('max')
print(df.max())
print('min')
print(df.min())

# df_ths = df[['t1','t2', 't3']]
# from pandas.plotting import table
# plt.figure(num=None, figsize=(9, 9), dpi=80, facecolor='w', edgecolor='k')
# ax = plt.subplot(111, frame_on=False) # no visible frame
# ax.xaxis.set_visible(False)  # hide the x axis
# ax.yaxis.set_visible(False)  # hide the y axis
#
# # table(ax, df, rowLabels=40, loc='center')  # where df is your data frame
# #
# # plt.savefig('mytable.png')
#
# df_ths.plot()
# plt.savefig('dense_depth_epochs_thresholds.png')
#
# df_other = df[['rel', 'rms', 'log10']]
#
# df_other.plot()
# plt.savefig('dense_depth_epochs_3_metrics.png')
