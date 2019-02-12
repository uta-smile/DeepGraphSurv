#
# from __future__ import print_function
# from __future__ import division
# from __future__ import unicode_literals

import numpy as np
import os
import sklearn.cluster as sc
import pickle
from multiprocessing import Pool


import graph_featurizer as gf
import graph_laplacian as gl
import io_utils

import glob


BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf')
DATA_DIR = os.path.join(BASE_DIR, 'data/WSI/TCGA_GBM_128/vgg_pca')
Train_dir_GBM = os.path.join(DATA_DIR, 'Train/')
Test_dir_GBM = os.path.join(DATA_DIR, 'Test/')

# Train_dir_LUSC = os.path.join(DATA_DIR, 'LUSC/vgg_pca/Train/')
# Test_dir_LUSC = os.path.join(DATA_DIR, 'LUSC/vgg_pca/Test/')
# assert os.path.exists(Train_dir_LUAD)
# assert os.path.exists(Test_dir_LUAD)
# assert os.path.exists(Train_dir_LUSC)
# assert os.path.exists(Test_dir_LUSC)

TRAIN_FILES_GBM = glob.glob(Train_dir_GBM+'*.np[yz]')
TEST_FILES_GBM = glob.glob(Test_dir_GBM+'*.npz')

# ALL_FILES = TRAIN_FILES + TEST_FILES
# TRAIN_FILES = map(lambda x: x.split('/')[-1], TRAIN_FILES)
# TEST_FILES = map(lambda x: x.split('/')[-1], TEST_FILES)

MIN_NUM_POINT = 4    # number of point you extract from raw files
MAX_NUM_POINT = 1000

TRAIN_NUM = 6000
TEST_NUM = 1024


def featurize(
              processed_data_dir,
              shard_size=1024,
              parallel=True):

    train_dir = os.path.join(processed_data_dir, 'train')
    test_dir = os.path.join(processed_data_dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    p = Pool(8)

    print("Loading and Processing Training Data......")
    metadata_rows = []
    all_X, all_y, all_L, all_names, all_size_x, all_node_img = [], [], [], [], [], []
    for shard_num, (shard, shard_y, shard_name, shard_nodes) in enumerate(get_shards([],
                                                                                     TRAIN_FILES_GBM,
                                                                                     shard_size,
                                                                                     True)):
        """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
            y is the class id for each sample (3D mesh), int type
        """
        print('Featurizing Training Data , Shard - %d' % shard_num)
        if parallel:
            # split jobs into pieces
            num_process = 8
            size_piece = int(len(shard) // num_process)
            job = []
            for id_pieces in range(num_process):
                start = id_pieces * size_piece
                end = (id_pieces + 1) * size_piece
                job.append(shard[start: end])
            if len(shard[end:]) > 0:
                job.append(shard[end:])

            # parallelly run them
            output = p.map(featurize_shard, job)

            # reunion the result
            shard_X, shard_L = [], []
            for piece in output:
                shard_X += piece[0]
                shard_L += piece[1]

        else:
            shard_X, shard_L = featurize_shard(shard)

        padded_X, padded_L = [], []
        size_x = []
        for x, l in zip(shard_X, shard_L):
            paded_x = np.lib.pad(x,
                                 ((0, MAX_NUM_POINT - x.shape[0]), (0, 0)),
                                 'constant',
                                 constant_values=(-1, -1))
            paded_l = np.lib.pad(l,
                                 ((0, MAX_NUM_POINT - l.shape[0]), (0, MAX_NUM_POINT - l.shape[1])),
                                 'constant',
                                 constant_values=(-1, -1))

            padded_X.append(paded_x)
            padded_L.append(paded_l)
            size_x.append(x.shape[0])

        padded_X = np.stack(padded_X).astype(np.float32)
        padded_L = np.stack(padded_L).astype(np.float32)
        size_x = np.squeeze(np.stack(size_x)).astype(np.int32)

        basename = "shard_{}".format(shard_num)
        """save to local"""
        metadata_rows.append(write_data_to_disk(train_dir,
                                                basename,
                                                padded_X,
                                                shard_y,
                                                padded_L,
                                                size_x,
                                                shard_name,
                                                shard_nodes))

        """ add up to list"""
        all_X.append(padded_X)
        all_L.append(padded_L)
        all_y.append(shard_y)
        all_names += shard_name
        all_size_x.append(size_x)
        all_node_img += shard_nodes

    """ return a Dataset contains all X, y, w, ids"""
    all_X = np.squeeze(np.vstack(all_X))
    all_L = np.squeeze(np.vstack(all_L))
    all_y = np.squeeze(np.concatenate(all_y)).astype(np.int32)
    all_size_x = np.squeeze(np.concatenate(all_size_x)).astype(np.int32)

    # "pad X and L"
    # max_atom_train = find_max_atom(all_L)
    # max_atom = max_atom_train

    """ save the meta data to local """
    meta_data1 = list()
    meta_data1.append(metadata_rows)
    meta_data1.append(MAX_NUM_POINT)
    with open(os.path.join(train_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_data1, f)

    # train_num = int(padded_X.shape[0] * 0.5)

    # create output dataset
    train_dataset = dict()
    train_dataset['X'] = all_X
    train_dataset['y'] = all_y
    train_dataset['names'] = all_names
    train_dataset['L'] = all_L
    train_dataset['size'] = all_size_x

    print("Loading and Processing Testing Data......")
    metadata_rows = []
    all_X, all_y, all_L, all_names, all_size_x, all_node_img = [], [], [], [], [], []
    for shard_num, (shard, shard_y, shard_name, shard_nodes) in enumerate(get_shards([],
                                                                                     TEST_FILES_GBM,
                                                                                     shard_size,
                                                                                     False)):
        """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
            y is the class id for each sample (3D mesh), int type
        """
        print('Featurizing Testing Data , Shard - %d' % shard_num)
        if parallel:
            # split jobs into pieces
            num_process = 8
            size_piece = int(len(shard) // num_process)
            job = []
            for id_pieces in range(num_process):
                start = id_pieces * size_piece
                end = (id_pieces + 1) * size_piece
                job.append(shard[start: end])
            if len(shard[end:]) > 0:
                job.append(shard[end:])
            # parallelly run them
            output = p.map(featurize_shard, job)

            # reunion the result
            shard_X, shard_L = [], []
            for piece in output:
                shard_X += piece[0]
                shard_L += piece[1]
        else:
            shard_X, shard_L = featurize_shard(shard)

        padded_X, padded_L = [], []
        size_x = []
        for x, l in zip(shard_X, shard_L):
            paded_x = np.lib.pad(x,
                                 ((0, MAX_NUM_POINT - x.shape[0]), (0, 0)),
                                 'constant',
                                 constant_values=(-1, -1))
            paded_l = np.lib.pad(l,
                                 ((0, MAX_NUM_POINT - l.shape[0]), (0, MAX_NUM_POINT - l.shape[1])),
                                 'constant',
                                 constant_values=(-1, -1))

            padded_X.append(paded_x)
            padded_L.append(paded_l)
            size_x.append(x.shape[0])

        padded_X = np.stack(padded_X).astype(np.float32)
        padded_L = np.stack(padded_L).astype(np.float32)
        size_x = np.squeeze(np.stack(size_x)).astype(np.int32)

        basename = "shard_{}".format(shard_num)
        """save to local"""
        metadata_rows.append(write_data_to_disk(test_dir,
                                                basename,
                                                padded_X,
                                                shard_y,
                                                padded_L,
                                                size_x,
                                                shard_name,
                                                shard_nodes))

        """ add up to list"""
        all_X.append(padded_X)
        all_L.append(padded_L)
        all_y.append(shard_y)
        all_names += shard_name
        all_size_x.append(size_x)
        all_node_img += shard_nodes

    "create label array"
    all_X = np.squeeze(np.vstack(all_X))
    all_L = np.squeeze(np.vstack(all_L))
    all_y = np.squeeze(np.concatenate(all_y)).astype(np.int32)
    all_size_x = np.squeeze(np.concatenate(all_size_x)).astype(np.int32)

    # "pad X and L"
    # max_atom_test = find_max_atom(all_L)
    # max_atom = max(max_atom_test, max_atom_train)

    """ save the meta data to local """
    meta_data2 = list()
    meta_data2.append(metadata_rows)
    meta_data2.append(MAX_NUM_POINT)
    with open(os.path.join(test_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_data2, f)

    """ return a Dataset contains all X, y, w, ids"""
    # create output dataset
    test_dataset = dict()
    test_dataset['X'] = all_X
    test_dataset['y'] = all_y
    test_dataset['names'] = all_names
    test_dataset['L'] = all_L
    test_dataset['size'] = all_size_x

    return train_dataset, test_dataset, MAX_NUM_POINT


def write_data_to_disk(
        data_dir,
        basename,
        X,
        y,
        L,
        size,
        names,
        node_img):
    """
    Write data to local as joblib format

    """
    out_X = "%s-X.joblib" % basename
    io_utils.save_to_disk(X, os.path.join(data_dir, out_X))

    out_y = "%s-y.joblib" % basename
    io_utils.save_to_disk(y, os.path.join(data_dir, out_y))

    out_L = "%s-L.joblib" % basename
    io_utils.save_to_disk(L, os.path.join(data_dir, out_L))

    out_size = "%s-size.joblib" % basename
    io_utils.save_to_disk(size, os.path.join(data_dir, out_size))

    out_names = "%s-name.joblib" % basename
    io_utils.save_to_disk(names, os.path.join(data_dir, out_names))

    out_node_img = "%s-node_img.joblib" % basename
    io_utils.save_to_disk(node_img, os.path.join(data_dir, out_node_img))

    # note that this corresponds to the _construct_metadata column order
    return {'basename': basename, 'X': out_X,
            'y': out_y, 'name': out_names,
            'L': out_L, 'size': out_size,
            'node_img': out_node_img}


def get_shards(data_dir, file_list, shard_size, istraining):

    """ shuffle the data files"""
    file_idxs = np.arange(0, len(file_list))
    np.random.shuffle(file_idxs)  # randomly extract data from files

    shard_num = len(file_list) // shard_size

    for shard_idx in range(shard_num):

        start_idx = shard_idx * shard_size
        end_idx = (shard_idx + 1) * shard_size
        shard_files_idxs = file_idxs[start_idx: end_idx]

        all_data, all_label, all_names, all_node_img = [], [], [], []
        for fn in shard_files_idxs:

            if not data_dir:
                raw_data = np.load(file_list[fn])
            else:
                raw_data = np.load(os.path.join(data_dir, file_list[fn]))

            current_data = raw_data['vgg_features']
            node_img_path = raw_data['img_path']
            # pid = raw_data['pid']
            # time = raw_data['time']
            if len(current_data) < MIN_NUM_POINT:
                # skip WSI of too few patches
                continue

            # if len(current_data) > MAX_NUM_POINT:
            #     continue

            curr_path = file_list[fn]

            curr_type = curr_path.split('/')[-4]
            curr_filename = curr_path.split('/')[-1]

            if curr_type == 'LUAD':
                # LUAD -> class 0, LUSC -> class 1
                current_label = 0
            else:
                current_label = 1

            # if istraining:
            "random select at most MAX_NUM_POINT nodes for WSI"
            list_node_idx = np.arange(0, current_data.shape[0])
            np.random.shuffle(list_node_idx)
            sel_ids = list_node_idx[0: MAX_NUM_POINT]

            current_data = current_data[sel_ids]
            current_data = np.expand_dims(current_data, 0)
            node_img_path = node_img_path[sel_ids]

            all_data.append(current_data)
            all_label.append(current_label)
            all_names.append(curr_filename)
            all_node_img.append(node_img_path)

        """ create numpy for all data and label"""
        all_label = np.squeeze(np.hstack(all_label))

        yield all_data, all_label, all_names, all_node_img


def find_max_atom(dataset):
    # find the maximum atom number in whole datasests
    max_n = 0
    for elm in dataset:
        max_n = max(elm.shape[0], max_n)
    return max_n


def featurize_shard(shard):

    """
    convert ndarray (n-sample, n_point of sample, 3-d)
    :param shard: ndarray, point cloud raw data
    :return: graph object for each sample
    """
    X = []  # to save the graph object
    L = []
    n_samples = len(shard)
    for idx in range(n_samples):
        print("processing sample %s\n" % str(idx))
        # iterate each pc in shard
        P = np.squeeze(np.array(shard[idx]))

        # do not need a clustering, use original points
        node_features = P
        adj_list, adj_matrix = gl.get_adjacency(node_features)
        Laplacian = gl.compute_laplacian(adj_list, normalized=False)
        X.append(node_features)
        L.append(Laplacian)

    return X, L


def load_back_from_disk(data_dir, istrain=True):
    """
    load data backas Train/test from disk
    :return: Train/Test STDiskDataset
    """
    """load back metadata_df"""
    meta_data = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
    metadata_rows = meta_data[0]
    max_node = meta_data[1]

    """itershard by loading from disk"""
    all_X, all_y, all_size, all_L, all_names, all_node_img = [], [], [], [], [], []

    for _, row in enumerate(metadata_rows):
        X = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['X'])))
        L = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['L'])))
        y = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['y'])))
        size = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['size'])))
        names = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['name'])))
        node_img = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['node_img'])))

        """ stack to list"""
        all_X.append(X)
        all_y.append(y)
        all_L.append(L)
        all_size.append(size)
        all_names.append(names)
        all_node_img.append(node_img)

    """ return a Dataset contains all X, y, w, ids"""
    all_X = np.squeeze(np.vstack(all_X))
    all_L = np.squeeze(np.vstack(all_L))
    all_y = np.squeeze(np.concatenate(all_y))
    all_size = np.squeeze(np.concatenate(all_size))
    all_names = np.squeeze(np.concatenate(all_names))
    all_node_img = np.squeeze(np.concatenate(all_node_img))

    # create output dataset
    dataset = dict()
    if istrain:
        dataset['X'] = all_X[:TRAIN_NUM]
        dataset['y'] = all_y[:TRAIN_NUM]
        dataset['size'] = all_size[:TRAIN_NUM]
        dataset['L'] = all_L[:TRAIN_NUM]
        dataset['name'] = all_names[:TRAIN_NUM]
        dataset['node_img'] = all_node_img[:TRAIN_NUM]
    else:
        dataset['X'] = all_X[:TEST_NUM]
        dataset['y'] = all_y[:TEST_NUM]
        dataset['size'] = all_size[:TEST_NUM]
        dataset['L'] = all_L[:TEST_NUM]
        dataset['name'] = all_names[:TEST_NUM]
        dataset['node_img'] = all_node_img[:TEST_NUM]

    return dataset, max_node


def load(dataset_name='processed_500maxnode'):

    """Load chemical datasets. Raw data is given as SMILES format"""
    # data_dir = os.path.join('../data/modelnet40_ply_hdf5_2048')
    # assert os.path.exists(data_dir)

    processed_data_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    if len(os.listdir(processed_data_dir)) != 0:

        # print("Loading Saved Data from Disk.......")

        """ pre-defined location for saving the train and test data"""
        train_dir = os.path.join(processed_data_dir, 'train')
        test_dir = os.path.join(processed_data_dir, 'test')

        train, max_node = load_back_from_disk(data_dir=train_dir, istrain=True)
        test, max_node_test = load_back_from_disk(data_dir=test_dir, istrain=False)
        max_node = max(max_node, max_node_test)

    else:
        train, test, max_node = featurize(
            processed_data_dir,
            shard_size=16)

    return train, test, max_node


def main():
    load('processed_{}maxnode'.format(MAX_NUM_POINT))


if __name__ == "__main__":
    main()
