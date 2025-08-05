#nii文件转npy文件，一个nii文件输出为一个文件夹的npy文件。
#sequence2oneFolder函数的作用是将子文件夹的npy文件转到一个文件夹下。
#rename_npy_filename用于重命名npy文件名。

from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import numpy as np
import matplotlib
import os
import cv2
import PIL
import shutil


def show_nii(file_path):
    # matplotlib.use('TkAgg')
    # 需要查看的nii文件名文件名.nii或nii.gz

    img = nib.load(file_path)
    # 打印文件信息
    #     print('img:', img)
    #     print('img.dataobj.shape:', img.dataobj.shape)
    width, height, queue = img.dataobj.shape
    print('width,height,queue:', width, height, queue)

    # 显示3D图像
    #     OrthoSlicer3D(img.dataobj).show()
    # # 计算看需要多少个位置来放切片图
    #     interval = 1
    #     x = int((queue/interval) ** 0.5) + 1
    #     num = 1
    #     plt.figure(figsize=(12, 12))
    i = 74
    img_arr = img.dataobj[:, :, i - 1]
    print(type(img_arr))
    img_arr = np.transpose(img_arr, (1, 0))
    print(img_arr.shape)
    print(img_arr[547 - 1][639 - 1], len(img_arr[415]))
    print(img_arr[4][639 - 1], len(img_arr[415]))
    # plt.axis('off')  # 去掉坐标轴
    plt.title('num:' + str(i))
    plt.imshow(img_arr, cmap='gray')
    plt.show()

# file_path = '/data/dwl/fetal_heart/try/data/1_nii_sequ_mask/211102-22+4w-z-dwl.nii.gz'
# show_nii(file_path)

def sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path):
    for nii_file in os.listdir(sequence_nii_path):
        print('nii_file name: ', nii_file)
        nii_file_path = os.path.join(sequence_nii_path, nii_file)

        nii_file_prefix = os.path.splitext(os.path.splitext(nii_file)[0])[0]
        sub_sequence_mask_npy_path = sequence_mask_npy_path + nii_file_prefix + '/'
        if os.path.isdir(sub_sequence_mask_npy_path) is False:
            os.mkdir(sub_sequence_mask_npy_path)
        else:
            continue
        try:
            mask = nib.load(nii_file_path)
        except nib.filebasedimages.ImageFileError as e:
            print('wrong nii_file name: ', nii_file)
            print(e)
            shutil.rmtree(sub_sequence_mask_npy_path)
            continue

        mask_arr = mask.dataobj
        width, height, deep = mask_arr.shape

        for step in range(deep):
            alone_mask_arr = mask_arr[:,:,step]
            alone_mask_arr = np.transpose(alone_mask_arr, (1,0))
            alone_mask_arr = np.array(alone_mask_arr,dtype=np.int16)
            np.save(sub_sequence_mask_npy_path + nii_file_prefix + f'-{step+1}.npy', alone_mask_arr)

def rename_npy_filename(alone_mask_npy_path):
    for alone_mask_npy_file in os.listdir(alone_mask_npy_path):
        alone_mask_npy_file_split = alone_mask_npy_file.split('-')
        if len(alone_mask_npy_file_split) == 5:
            new_alone_mask_npy_file = alone_mask_npy_file_split[0] + '-' + alone_mask_npy_file_split[1] + '-' + alone_mask_npy_file_split[2] + '-' + alone_mask_npy_file_split[4]
        else:
            new_alone_mask_npy_file = alone_mask_npy_file_split[0] + '-' + alone_mask_npy_file_split[1] + '-' + alone_mask_npy_file_split[3]
        alone_mask_npy_file_path = os.path.join(alone_mask_npy_path, alone_mask_npy_file)
        new_alone_mask_npy_file = os.path.join(alone_mask_npy_path, new_alone_mask_npy_file)
        os.rename(alone_mask_npy_file_path, new_alone_mask_npy_file)

if __name__ == "__main__":
    #二腔心切面，包括健康和不健康
    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_nii_LA_final/'
    sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_npy_LA_final/'
    sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_nii_LA/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_npy_LA/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # 三腔心切面，包括健康和不健康
    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_nii_LA/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy_LA/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)
    # print()
    # print()

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_nii_LA_final/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy_LA_final/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)
    # print()
    # print()
    #
    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_nii_LA/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_npy_LA/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # 四腔心切面，包括健康和不健康
    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)
    # print()
    # print()

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_nii_LA/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy_LA/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_nii_LA_final/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy_LA_final/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)
    #
    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_nii_LA/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_npy_LA/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # 短轴切面，包括健康和不健康
    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_mask_nii_WM/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_mask_npy_WM/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_mask_nii/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_mask_npy/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)

    # sequence_nii_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_mask_nii_WM/'
    # sequence_mask_npy_path = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_mask_npy_WM/'
    # sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)