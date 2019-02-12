#!/usr/bin/env python
'''
This is to load dataset from npz files
MICCAI 2018 preparation

Use VGG16 to extract features

'''

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
from keras.layers import merge, Dense, Input, Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.models import Sequential, Model
import numpy as np
import pandas as pd
# import cv2
import scipy.misc as sci
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedShuffleSplit
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
import os
from keras.utils import multi_gpu_model

import tensorflow as tf
import random
import glob


def return_Dict_pos(sel_img_name):
    """
    Get x, y pos of patches from WSI

    :param sel_img_name: Given patch name
    :return: position of the patch from WSI
    """
    pos_list = np.zeros((len(sel_img_name),2))

    for ii, img_name in enumerate(sel_img_name):


        if 'rpos' in img_name:
            r_pos = img_name.split('-rpos')[-1]
        else:
            r_pos = img_name.split('-r')[-1]
        r_pos = r_pos.split('-')[0]


        if 'cpos' in img_name:
            c_pos = img_name.split('-cpos')[-1]
        else:
            c_pos = img_name.split('-c')[-1]
        c_pos = c_pos.split('-')[0]
        pos_list[ii][0] = int(r_pos)
        pos_list[ii][1] = int(c_pos)

    c_min = min(pos_list[:,1])
    c_max = max(pos_list[:,1])

    r_min = min(pos_list[:,0])
    r_max = max(pos_list[:,0])

    zero_image = sel_img_name[0]

    width_string = zero_image.split('-')[-1]
    width = int(width_string.split('.')[0])

    ori_width = (r_max + width - r_min) / width
    ori_length = (c_max + width - c_min) / width

    curr_r = (pos_list[:,0] - r_min) / width
    curr_c = (pos_list[:,1] - c_min) / width

    final_pos_1d = curr_c * ori_width + curr_r

    sorted_pos = np.sort(final_pos_1d)

    # ind_c = np.argsort(pos_list, 1)
    #
    unique_sortc = np.argsort(final_pos_1d)

    sort_img_name = [sel_img_name[i] for i in unique_sortc]

    return unique_sortc


def get_vgg_features(base_path, dataset, file_name, usePCA=True):
    """
    This is to extract vgg features from Patches

    :param base_path: Loading path
    :param file_name:
    :param usePCA: if use PCA for clustering
    :return:
    """
    # base_path = '/home/jy/Data_JY/Patches/Shanghai/1/10x/'
    path = os.path.join(base_path)

    with tf.device('/cpu:0'):
        base_model = VGG16(weights='imagenet', include_top=True)
        model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    parallel_model = multi_gpu_model(model, gpus=2)
    # base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
    # input = Input(shape=(224, 224, 3), name='image_input')
    # x = base_model(input)
    # x = Flatten()(x)
    # model = Model(inputs=input, outputs=x)
    data_set = '_'.join(file_name.split('_')[:2])
    isTrain = file_name.split('_')[2]
    patch_files = pd.read_csv(file_name)

    # pid = map(int, patch_files["pid"])
    if data_set in ('NLST_SCC','NLST_ADC'):
        pid = map(str, patch_files["wsi_name"])
    else:
        pid = map(str, patch_files["pid"])
    time = map(float, patch_files["survival"])
    status = map(int, patch_files["status"])


    for r, d, f in os.walk(path):
        folder = r.split("/")[-1]
        # print folder
        if folder:
            a = folder.split("-")[:3]
            pid_temp = "-".join(a)

            if data_set in ('NLST_SCC','NLST_ADC'):
                curr_patient = pid_temp + '.svs'
            else:
                curr_patient = pid_temp

            look_up_name = curr_patient

            if look_up_name not in pid:
                continue

            curr_img_files = glob.glob(os.path.join(base_path, folder) + '/*.tiff')

            if curr_img_files:
                ind_sort_img_name = return_Dict_pos(curr_img_files)

                img_batch = []

                sort_img_path = [curr_img_files[i] for i in ind_sort_img_name]

                save_img_path = []

                for i, img_id in enumerate(sort_img_path):
                    curr_path = img_id
                    img = cv2.imread(curr_path)
                    temp = curr_path.split('/')[-1]
                    temp_path = temp.split('-')[-3:]
                    save_img_path.append("-".join(temp_path))
                    # status_empty = ischeck_empty(img)
                    #
                    # if status_empty:
                    #     continue
                    img_resize_x = sci.imresize(img, (224, 224, 3))
                    # img_resize_x = np.swapaxes(img_resize_x, 0, 2)
                    vgg_input_x = np.expand_dims(img_resize_x.astype('float32'), axis=0)
                    vgg_input_x = preprocess_input(vgg_input_x)
                    img_batch.append(vgg_input_x)

                stack_batch = np.vstack(img_batch)
                fea1 = parallel_model.predict(stack_batch)

                if usePCA:
                    fea_pca = fea1
                    # fea_pca = pca.transform(fea1)
                    save_folder = 'vgg_pca'
                else:
                    fea_pca = fea1
                    save_folder = 'vgg_4096'

                index = pid.index(look_up_name)
                save_path = '/'.join([data_set, save_folder, isTrain])+ '/' + str(folder) + '.npz'

                print look_up_name, time[index], status[index]
                np.savez(save_path, vgg_features=fea_pca,
                         pid=look_up_name, time=time[index], status=status[index], img_path=save_img_path)



def Train_PCA_From_NPZ(path, folder, data_set, out_dim=128):
    """
    Use 10% patches to train PCA on training data only
    :param path:
    :param folder:
    :param data_set:
    :param out_dim:
    :return:
    """
    import glob
    sel_files = []
    sel_idd = []

    for each_set in data_set:
        file_path = folder + each_set + '/' + path
        npz_files = glob.glob(file_path+'/*.npz')

        for idd, cur_file in enumerate(npz_files):
            all_fea = []
            img_id = []
            # if random.randint(0, len(npz_files)) < len(npz_files)/10:
            #     continue
            # if idd>10:
            #     break
            Train_vgg_file = np.load(cur_file)
            cur_vgg_features = Train_vgg_file['vgg_features']
            cur_pid = Train_vgg_file['pid']
            cur_img_path = Train_vgg_file['img_path']
            if len(cur_img_path)<2:
                continue
            all_fea.append(cur_vgg_features)
            for _ in xrange(len(cur_vgg_features)):
                img_id.append(idd)
            print cur_pid

            fea_batch = np.vstack(all_fea)
            img_id_batch = np.vstack(img_id)

            group_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
            for train_index, test_index in group_sss.split(fea_batch, img_id_batch):
                for i in test_index:
                    sel_files.append(fea_batch[i])
                    sel_idd.append(img_id[i])

            fea_batch = []
            img_id_batch = []

    train_pca_fea = np.vstack(sel_files)
    pca = PCA(n_components=out_dim, whiten=True).fit(train_pca_fea)

    # np.savez(data_set + '_PCA.npz', pca=pca)
    # print "pca finished"

    return pca


def extract_fea_TCGA(path, folder, pca, data_set, isTrain):
    """
    Use trained PCA for dimension reduction on both training and testing set
    :param path:
    :param folder:
    :param pca:
    :param data_set:
    :param isTrain:
    :return:
    """

    for TrainorTest in isTrain:
        for each_set in data_set:
            file_path = folder + each_set + '/' + path + TrainorTest
            npz_files = glob.glob(file_path + '/*.npz')
            for idd, cur_file in enumerate(npz_files):
        # if idd>10:
        #     break
                Train_vgg_file = np.load(cur_file)
                cur_vgg_features = Train_vgg_file['vgg_features']
                cur_pid = Train_vgg_file['pid']
                cur_time = Train_vgg_file['time']
                cur_status = Train_vgg_file['status']
                cur_img_path = Train_vgg_file['img_path']

                fea_pca = pca.transform(cur_vgg_features)
                save_path = folder +'/'.join([each_set, 'vgg_pca', TrainorTest]) + '/' + cur_file.split('/')[-1]
                np.savez(save_path, vgg_features=fea_pca, pid=cur_pid, time=cur_time, status=cur_status, img_path=cur_img_path)

                print save_path


if __name__ == "__main__":
    folder = '../dataset/'
    data_set = ['TCGA_GBM']
    path = 'vgg_4096/Train'

    pca_path = '../dataset/'
    isTrain = ['Train','Test']
    pca_path = 'vgg_4096/'

    # First train PCA from training npz, using about 10%
    pca = Train_PCA_From_NPZ(path, folder, data_set)

    # Extract PCA features on training and testing set
    extract_fea_TCGA(pca_path, folder, pca, data_set, isTrain)