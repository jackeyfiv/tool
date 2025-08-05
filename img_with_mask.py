#img和mask都是npy文件

import os
import shutil

import cv2
import numpy as np


def img_with_mask_1folder(alone_mask_npy2, alone_image_npy, image_with_mask):
    for image_npy in os.listdir(alone_image_npy):
        image_npy_prefix = os.path.splitext(image_npy)[0]
        image_npy_path = os.path.join(alone_image_npy, image_npy)
        image_npy_mat = np.load(image_npy_path)
        image_npy_mat = image_npy_mat[..., np.newaxis]
        image_npy_mat = np.array(image_npy_mat, dtype=np.float64)

        mask_npy_path = os.path.join(alone_mask_npy2, image_npy)
        mask_npy_mat = np.load(mask_npy_path)
        mask_npy_mat = mask_npy_mat / np.max(mask_npy_mat) * 255
        mask_npy_mat = mask_npy_mat[..., np.newaxis]
        #         mask_npy_mat = np.array(mask_npy_mat,dtype=np.float64)

        image_with_mask_mat = cv2.addWeighted(image_npy_mat, 1.0, mask_npy_mat, 0.5, 1)
        cv2.imwrite(image_with_mask + image_npy_prefix + '.png', image_with_mask_mat)


def img_with_mask_1folder_color(alone_mask_npy2, alone_image_npy, image_with_mask):
    for image_npy in os.listdir(alone_image_npy):
        image_npy_prefix = os.path.splitext(image_npy)[0]
        image_npy_path = os.path.join(alone_image_npy, image_npy)
        image_npy_mat = np.load(image_npy_path)
        #         image_npy_mat = image_npy_mat[...,np.newaxis]
        image_npy_mat = np.array(image_npy_mat, dtype=np.float64)
        image_npy_mat = np.tile(image_npy_mat, (3, 1, 1))
        image_npy_mat = np.swapaxes(image_npy_mat, 0, 1)
        image_npy_mat = np.swapaxes(image_npy_mat, 1, 2)

        mask_npy_path = os.path.join(alone_mask_npy2, image_npy)
        mask_npy_mat = np.load(mask_npy_path)
        mask_npy_mat = np.tile(mask_npy_mat, (3, 1, 1))
        mask_npy_mat = np.swapaxes(mask_npy_mat, 0, 1)
        mask_npy_mat = np.swapaxes(mask_npy_mat, 1, 2)

        index = np.where(mask_npy_mat == [1, 1, 1])
        for num0 in range(len(index[0])):
            mask_npy_mat[index[0][num0], index[1][num0], :] = [255.0, 0, 0]
        #             mask_npy_mat[index[0][num0], index[1][num0], index[2][0]] = 255.0
        #             mask_npy_mat[index[0][num0], index[1][num0], index[2][1]] = 0
        #             mask_npy_mat[index[0][num0], index[1][num0], index[2][2]] = 0

        index = np.where(mask_npy_mat == [2, 2, 2])
        for num0 in range(len(index[0])):
            mask_npy_mat[index[0][num0], index[1][num0], :] = [0, 255.0, 0]

        #         for row in range(mask_npy_mat.shape[0]):
        #             for col in range(mask_npy_mat.shape[1]):
        # #                 print(mask_npy_mat[row,col])
        #                 if (mask_npy_mat[row,col] == [1,1,1]).any():
        #                     mask_npy_mat[row,col] = [255,0,0]
        #                 elif (mask_npy_mat[row,col] == [2,2,2]).any():
        #                     mask_npy_mat[row,col] = [0,255,0]
        #                 else:
        #                     pass
        mask_npy_mat = np.array(mask_npy_mat, dtype=np.float64)
        image_with_mask_mat = cv2.addWeighted(image_npy_mat, 1.0, mask_npy_mat, 0.5, 1)
        cv2.imwrite(image_with_mask + image_npy_prefix + '.png', image_with_mask_mat)
#         break

def img_with_mask_2folder_color(sequence_mask_npy2, sequence_image_npy, sequence_image_with_mask, mask_name=None, scale=None):
    for subfolder in os.listdir(sequence_image_npy):
        subfolder_path = os.path.join(sequence_image_npy, subfolder) + '/'
        sub_sequence_image_with_mask = sequence_image_with_mask + subfolder
        if os.path.isdir(sub_sequence_image_with_mask) == False:
            os.mkdir(sub_sequence_image_with_mask)
        else:
            continue

        for image_npy in os.listdir(subfolder_path):
            image_npy_prefix = os.path.splitext(image_npy)[0]
            image_npy_path = os.path.join(subfolder_path, image_npy)
            image_npy_mat = np.load(image_npy_path)
            image_npy_shape = image_npy_mat.shape
            image_npy_mat = np.array(image_npy_mat, dtype=np.float64)

            if scale is not None:
                image_npy_mat = image_npy_mat[..., np.newaxis]
                # image_npy_mat = cv2.resize(image_npy_mat, (100, 100))
                image_npy_mat = cv2.resize(image_npy_mat, scale, interpolation=cv2.INTER_NEAREST)

            image_npy_mat = np.tile(image_npy_mat, (3, 1, 1))
            image_npy_mat = np.swapaxes(image_npy_mat, 0, 1)
            image_npy_mat = np.swapaxes(image_npy_mat, 1, 2)

            #             subfolder_split = subfolder.split('-')
            #             if len(subfolder_split) == 3:
            #                 new_subfolder = subfolder_split[0] + '-' + subfolder_split[1] + '-' + subfolder_split[2] + personName
            #             else:
            #                 new_subfolder = subfolder_split[0] + '-' + subfolder_split[1] + personName
            #             image_npy_split = image_npy.split('-')
            #             if len(image_npy_split) == 4:
            #                 new_image_npy = image_npy_split[0] + '-' + image_npy_split[1] + '-' + image_npy_split[2] + personName + '-' + image_npy_split[3]
            #             else:
            #                 new_image_npy = image_npy_split[0] + '-' + image_npy_split[1] + personName + '-' + image_npy_split[2]

            if mask_name is not None:
                mask_npy_folder = image_npy.split('-')[0] + '-' + mask_name
                mask_npy_path = sequence_mask_npy2 + mask_npy_folder + '/' + mask_npy_folder + '-' + image_npy.split('-')[2].split('.')[0] + '.npy'
            else:
                mask_npy_path = sequence_mask_npy2 + subfolder + '/' + image_npy

            try:
                mask_npy_mat = np.load(mask_npy_path)
            except (FileNotFoundError):
                print(mask_npy_path)
                if os.path.isdir(sub_sequence_image_with_mask) == True:
                    shutil.rmtree(sub_sequence_image_with_mask)
                continue
            mask_npy_shape = mask_npy_mat.shape

            if image_npy_shape != mask_npy_shape:
                print(mask_npy_path, 'the shape of image and mask is different!!!!!!!!!!!!!!!!!!')

            if scale is not None:
                mask_npy_mat = mask_npy_mat[..., np.newaxis]
                # image_npy_mat = cv2.resize(image_npy_mat, (100, 100))
                mask_npy_mat = cv2.resize(mask_npy_mat, scale, interpolation=cv2.INTER_NEAREST)

            mask_npy_mat = np.tile(mask_npy_mat, (3, 1, 1))
            mask_npy_mat = np.swapaxes(mask_npy_mat, 0, 1)
            mask_npy_mat = np.swapaxes(mask_npy_mat, 1, 2)

            index = np.where(mask_npy_mat == [1, 1, 1])
            for num0 in range(len(index[0])):
                mask_npy_mat[index[0][num0], index[1][num0], :] = [255.0, 0, 0]
            index = np.where(mask_npy_mat == [2, 2, 2])
            for num0 in range(len(index[0])):
                mask_npy_mat[index[0][num0], index[1][num0], :] = [0, 255.0, 0]
            index = np.where(mask_npy_mat == [3, 3, 3])
            for num0 in range(len(index[0])):
                mask_npy_mat[index[0][num0], index[1][num0], :] = [0, 0, 255.0]
            mask_npy_mat = np.array(mask_npy_mat, dtype=np.float64)

            image_with_mask_mat = cv2.addWeighted(image_npy_mat, 1.0, mask_npy_mat, 0.5, 1)
            cv2.imwrite(sub_sequence_image_with_mask + '/' + image_npy_prefix + '.png', image_with_mask_mat)

# img_with_mask_subfolder_color(sequence_mask_npy2, sequence_image_npy, sequence_image_with_mask_color)
# personName 类似于是'-' + 'dwl'
def img_with_mask_2folder(sequence_mask_npy2, sequence_image_npy, sequence_image_with_mask, personName=''):
    for subfolder in os.listdir(sequence_image_npy):
        subfolder_path = os.path.join(sequence_image_npy, subfolder) + '/'
        sub_sequence_image_with_mask = sequence_image_with_mask + subfolder
        if os.path.isdir(sub_sequence_image_with_mask) == False:
            os.mkdir(sub_sequence_image_with_mask)
        for image_npy in os.listdir(subfolder_path):
            image_npy_prefix = os.path.splitext(image_npy)[0]
            image_npy_path = os.path.join(subfolder_path, image_npy)
            image_npy_mat = np.load(image_npy_path)
            image_npy_mat = image_npy_mat[..., np.newaxis]
            image_npy_mat = np.array(image_npy_mat, dtype=np.float64)

            subfolder_split = subfolder.split('-')
            if len(subfolder_split) == 3:
                new_subfolder = subfolder_split[0] + '-' + subfolder_split[1] + '-' + subfolder_split[2] + personName
            else:
                new_subfolder = subfolder_split[0] + '-' + subfolder_split[1] + personName

            image_npy_split = image_npy.split('-')
            #             print(image_npy)
            if len(image_npy_split) == 4:
                new_image_npy = image_npy_split[0] + '-' + image_npy_split[1] + '-' + image_npy_split[
                    2] + personName + '-' + image_npy_split[3]
            else:
                new_image_npy = image_npy_split[0] + '-' + image_npy_split[1] + '-' + personName + image_npy_split[2]

            mask_npy_path = sequence_mask_npy2 + new_subfolder + '/' + new_image_npy
            try:
                mask_npy_mat = np.load(mask_npy_path)
            except (FileNotFoundError):
                print(mask_npy_path)
                continue
            mask_npy_mat = mask_npy_mat / np.max(mask_npy_mat) * 255
            mask_npy_mat = mask_npy_mat[..., np.newaxis]

            image_with_mask_mat = cv2.addWeighted(image_npy_mat, 1.0, mask_npy_mat, 0.5, 1)
            #             print(image_with_mask_mat.shape)
            cv2.imwrite(sub_sequence_image_with_mask + '/' + image_npy_prefix + '.png', image_with_mask_mat)

if __name__ == '__main__':
    #二腔心切面，包括健康组和非健康组
    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_npy_LA/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_image_with_mask_LA/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='2cavityf', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='A2CV', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_npy_LA/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_image_with_mask_LA/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='A2CA', scale=(100,100))

    #三腔心切面，包括健康组和非健康组
    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy_LA/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_image_with_mask_LA/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='3cavityf', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='A3CV', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_npy_LA/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_image_with_mask_LA/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='A3CA', scale=(100,100))

    #四腔心切面，包括健康组和非健康组
    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy_LA/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_image_with_mask_LA/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='4cavityf', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='A4CV', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_npy_LA/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_image_with_mask_LA/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='A4CA', scale=(100,100))

    # 短轴平面，包括健康组和非健康组
    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='LV short', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_mask_npy_WM/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_image_with_mask_WM/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='out', scale=(100,100))

    # sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_mask_npy/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_image_npy/'
    # sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_image_with_mask/'
    # img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='NM', scale=(100,100))

    sequence_mask_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_mask_npy_WM/'
    sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_image_npy/'
    sequence_image_with_mask = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_image_with_mask_WM/'
    img_with_mask_2folder_color(sequence_mask_npy, sequence_image_npy, sequence_image_with_mask, mask_name='WM', scale=(100,100))