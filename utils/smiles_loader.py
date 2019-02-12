
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import sklearn.cluster as sc
from multiprocessing import Pool
import gzip
from matplotlib import pyplot
import matplotlib as plt
import pickle
import pandas as pd
import numpy as np
from random import shuffle

from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit import Chem


BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import graph_featurizer as gf
import graph_laplacian as gl
import io_utils

TEST_RATIO = 0.2


def featurize(data_dir,
              file_name,
              processed_data_dir,
              shard_size=1024,
              parallel=False,
              useGraph=True):

    train_dir = os.path.join(processed_data_dir, 'train')
    test_dir = os.path.join(processed_data_dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    p = Pool(8)

    print("Loading and Processing Training Data......")
    metadata_rows, metadata_rows2 = [], []
    all_X, all_y, all_L, all_adj = [], [], [], []

    smile_col = 'smiles'
    label_col = 'measured log solubility in mols per litre'

    data_path = os.path.join(data_dir, file_name)
    for shard_num, (shard, shard_y) in enumerate(get_shards(
            data_path,
            smile_col,
            label_col,
            shard_size)):
        """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
            y is the class id for each sample (3D mesh), int type
        """
        print('Featurizing Training Data , Shard - %d' % shard_num)
        if parallel:
            # split data into jobs
            num_process = 8
            size_piece = int(len(shard) // num_process)
            job = []
            for id_pieces in range(num_process):
                start = id_pieces * size_piece
                end = (id_pieces + 1) * size_piece
                job.append(shard[start: end])
            if len(shard[end:]) > 0:
                # all rest data got to the last job
                job.append(shard[end:])

            # run them in parallel way
            output = p.map(featurize_shard, job)

            # reunion the result
            X, L = [], []
            for piece in output:
                X.append(piece[0])
                L.append(piece[1])
            shard_X = np.vstack(X)
            shard_L = np.vstack(L)
        else:
            # put all shard into it as one job
            shard_X, shard_y, shard_L, shard_adj = featurize_shard(shard, shard_y)

        """ stack to list of shard"""
        all_X += shard_X
        all_y += shard_y
        all_L += shard_L
        all_adj += shard_adj

    "get the max atom of compound in this dataset"
    max_atom = find_max_atom(all_L)
    padded_X, padded_L = [], []
    size_x = []
    for x, l in zip(all_X, all_L):
        paded_x = np.lib.pad(x,
                             ((0, max_atom - x.shape[0]), (0, 0)),
                             'constant',
                             constant_values=(-1, -1))
        paded_l = np.lib.pad(l,
                             ((0, max_atom - l.shape[0]), (0, max_atom - l.shape[1])),
                             'constant',
                             constant_values=(-1, -1))

        padded_X.append(paded_x)
        padded_L.append(paded_l)
        size_x.append(x.shape[0])

    padded_X = np.stack(padded_X).astype(np.float32)
    padded_L = np.stack(padded_L).astype(np.float32)
    all_y = np.stack(all_y).astype(np.float32)
    size_x = np.squeeze(np.stack(size_x)).astype(np.int32)

    test_num = int(len(padded_L) * TEST_RATIO)
    shuffled_idx = np.arange(padded_L.shape[0]).tolist()
    np.random.shuffle(shuffled_idx)

    train_X = padded_X[shuffled_idx[test_num:]]
    train_L = padded_L[shuffled_idx[test_num:]]
    train_y = all_y[shuffled_idx[test_num:]]
    train_size_x = size_x[shuffled_idx[test_num:]]

    train_adj = [all_adj[idx] for idx in shuffled_idx[test_num:]]

    test_X = padded_X[shuffled_idx[:test_num]]
    test_L = padded_L[shuffled_idx[:test_num]]
    test_y = all_y[shuffled_idx[:test_num]]
    test_size_x = size_x[shuffled_idx[:test_num]]
    test_adj = [all_adj[idx] for idx in shuffled_idx[:test_num]]

    """ save shard (x,y, L)"""
    basename = "shard-%d" % 0
    metadata_rows.append(write_data_to_disk(train_dir, basename,
                                            train_X, train_y, train_L, train_size_x, train_adj))
    metadata_rows2.append(write_data_to_disk(test_dir, basename,
                                            test_X, test_y, test_L, test_size_x, test_adj))
    ""

    """ save the meta data to local """
    meta_data1 = list()
    meta_data1.append(metadata_rows)
    meta_data1.append(max_atom)
    with open(os.path.join(train_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_data1, f)

    meta_data2 = list()
    meta_data2.append(metadata_rows2)
    meta_data2.append(max_atom)
    with open(os.path.join(test_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_data2, f)

    # create output dataset
    train_dataset = dict()
    train_dataset['X'] = train_X
    train_dataset['y'] = train_y
    train_dataset['L'] = train_L
    train_dataset['size'] = train_size_x
    train_dataset['adj_list'] = train_adj

    test_dataset = dict()
    test_dataset['X'] = test_X
    test_dataset['y'] = test_y
    test_dataset['L'] = test_L
    test_dataset['size'] = test_size_x
    test_dataset['adj_list'] = test_adj

    return train_dataset, test_dataset, max_atom


def write_data_to_disk(
        data_dir,
        basename,
        X,
        y,
        L,
        size,
        adj):
    """
    Write data to local as joblib format

    """
    out_X = "%s-X.joblib" % basename
    io_utils.save_to_disk(X, os.path.join(data_dir, out_X))

    out_y = "%s-y.joblib" % basename
    io_utils.save_to_disk(y, os.path.join(data_dir, out_y))

    out_L = "%s-L.joblib" % basename
    io_utils.save_to_disk(L, os.path.join(data_dir, out_L))

    out_adj = "%s-adj.joblib" % basename
    io_utils.save_to_disk(adj, os.path.join(data_dir, out_adj))

    out_size = "%s-size.joblib" % basename
    io_utils.save_to_disk(size, os.path.join(data_dir, out_size))

    # note that this corresponds to the _construct_metadata column order
    return {'basename': basename, 'X': out_X, 'y': out_y,
            'L': out_L, 'size': out_size, 'adj_list': out_adj}


def get_shards(data_dir, data_col, label_col, shard_size):
    return load_csv_files(data_dir, data_col, label_col, shard_size)


def find_max_atom(dataset):
    # find the maximum atom number in whole datasests
    max_n = 0
    for elm in dataset:
        max_n = max(elm.shape[0], max_n)
    return max_n


def featurize_shard(shard, shard_y):
    """Featurize a shard of an input dataframe."""
    return featurize_smiles_df(shard, shard_y)


def featurize_smiles_df(shard, shard_y):
    """
    convert ndarray (n-sample, n_point of sample, 3-d)
    """

    # generate Laplacian by nodes raw feature (intensity + coordinates)

    """Featurize individual compounds in dataframe.
   Given a featurizer that operates on individual chemical compounds
   or macromolecules, compute & add features for that compound to the
   features dataframe
    """

    features_list = []
    laplacian_list = []
    adjacency_list = []
    sel_y = []
    for ind, elem in enumerate(shard):
        mol = Chem.MolFromSmiles(elem)

        if mol:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        node_feature, canon_adj_list = gf.get_graph_featurize(mol)
        if node_feature.shape[0] < 5:
            continue
        features_list.append(node_feature)

        laplacian = gl.compute_laplacian(canon_adj_list, normalized=False)
        laplacian_list.append(laplacian)
        adjacency_list.append(canon_adj_list)
        sel_y.append(shard_y[ind])

    valid_inds = np.array([1 if elt.shape[0] > 1 else 0 for elt in features_list], dtype=bool)
    features = [elt for (is_valid, elt) in zip(valid_inds, features_list) if is_valid]
    sel_y = [elt for (is_valid, elt) in zip(valid_inds, sel_y) if is_valid]
    laplacian = [elt for (is_valid, elt) in zip(valid_inds, laplacian_list) if is_valid]
    adjacency_list = [elt for (is_valid, elt) in zip(valid_inds, adjacency_list) if is_valid]

    assert len(features) == len(laplacian) == len(adjacency_list)
    return features, sel_y, laplacian, adjacency_list


def load_back_from_disk(data_dir, useGraph=True):
    """load back from metadata_df"""
    meta_data = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
    metadata_rows = meta_data[0]
    max_node = meta_data[1]

    """itershard by loading from disk"""
    all_X, all_y, all_L, all_size_x, all_adj = [], [], [], [], []

    for _, row in enumerate(metadata_rows):
        X = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['X'])))
        y = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['y'])))
        if useGraph:
            L = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['L'])))
            all_L.append(L)
        size_x = np.array(io_utils.load_from_disk(os.path.join(data_dir, row['size'])))
        adj_list = io_utils.load_from_disk(os.path.join(data_dir, row['adj_list']))

        """ stack to list"""
        all_X.append(X)
        all_y.append(y)
        all_size_x.append(size_x)
        all_adj += adj_list     # list

    """ return a Dataset contains all X, y, w, L"""

    all_X = np.squeeze(np.vstack(all_X))
    all_y = np.squeeze(np.concatenate(all_y))
    all_size_x = np.squeeze(np.concatenate(all_size_x))
    if useGraph:
        all_L = np.squeeze(np.vstack(all_L))

    # create output dataset
    dataset = dict()
    dataset['X'] = all_X
    dataset['y'] = all_y
    dataset['L'] = all_L
    dataset['size'] = all_size_x
    dataset['adj_list'] = all_adj

    return dataset, max_node


def load_csv_files(filenames_list, data_col, label_col, shard_size=None, verbose=True):
    """Load data as pandas dataframe."""
    # First line of user-specified CSV *must* be header.
    shard_num = 1
    if type(filenames_list) is not list:
        filenames_list = [filenames_list]
    for filename in filenames_list:
        if shard_size is None:
            yield pd.read_csv(filename)
        else:
            for df in pd.read_csv(filename, chunksize=shard_size):

                df = df.replace(np.nan, str(""), regex=True)
                smiles = df[data_col].tolist()
                labels = df[label_col].tolist()
                shard_num += 1
                yield smiles, labels


def load(data_folder='delaney',
         file_name='delaney-processed.csv',
         useGraph=True):
    package_dir = os.path.join(os.environ["HOME"], 'AGCN_tf/')
    data_dir = os.path.join(package_dir, 'data/chemistry', data_folder)
    assert os.path.exists(data_dir)

    """decide use laplacian or not"""

    if useGraph:
        processed_data_dir = os.path.join(data_dir, 'processed_data')
    else:
        processed_data_dir = os.path.join(data_dir, 'original_images')

    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    if len(os.listdir(processed_data_dir)) != 0:
        print("Loading Saved Data from Disk.......")

        """ pre-defined location for saving the train and test data"""
        train_dir = os.path.join(processed_data_dir, 'train')
        test_dir = os.path.join(processed_data_dir, 'test')

        train, max_node_train = load_back_from_disk(data_dir=train_dir, useGraph=useGraph)
        test, max_node_test = load_back_from_disk(data_dir=test_dir, useGraph=useGraph)

        assert max_node_test == max_node_train
        max_node = max(max_node_test, max_node_train)

    else:
        print("Loading and featurizing data.......")

        train, test, max_node = featurize(
            data_dir,
            file_name,
            processed_data_dir,
            shard_size=1024,
            useGraph=useGraph)

    return train, test, max_node


def main():
    load_smiles()


if __name__ == "__main__":
    main()
