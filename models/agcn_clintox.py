import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import networkx as nx
from sklearn.metrics import mean_squared_error
from math import sqrt

BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import smiles_loader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='AGCN',
                    help='Model name: AGCN[default: AGCN]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=55, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=2e5, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
MAX_NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_FEATURE = 75
NUM_CLASSES = 40


MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

IMG_DIR = os.path.join(LOG_DIR, 'images2')
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

# hyper-parameters

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-8)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            data_pl, labels_pl, laplacian_pl, size_x_pl = MODEL.placeholder_inputs(BATCH_SIZE, MAX_NUM_POINT, NUM_FEATURE)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.basic_agcn_smile(data_pl,
                                                      laplacian_pl,
                                                      size_x_pl,
                                                      MAX_NUM_POINT,
                                                      1,
                                                      is_training_pl,
                                                      bn_decay=bn_decay)
            loss = MODEL.mse(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            reg_loss = MODEL.loss_reg(end_points['regs_list'])
            # Get training operator
            # learning_rate = get_learning_rate(batch)
            # tf.summary.scalar('learning_rate', learning_rate)

            learning_rate = tf.placeholder(tf.float32, shape=())

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss + 0.01 * reg_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': data_pl,
               'labels_pl': labels_pl,
               'laplacian_pl': laplacian_pl,
               'size_x_pl': size_x_pl,
               'is_training_pl': is_training_pl,
               'laplacian': end_points['lap'],
               'res_lap': end_points['res_lap'],
               'pred': pred,
               'loss': loss,
               'lr': learning_rate,
               'train_op': train_op,
               'step': batch}
        loss_seq = []

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            loss_seq = train_one_epoch(epoch, sess, ops, loss_seq)

            if epoch % 5 == 0 and epoch > 0:
                evaluate_one_epoch(epoch, sess, ops)

        # plot_loss(loss_seq)


def plot_loss(epoch, loss_seq):
    import matplotlib.pyplot as plt
    plt.figure(3)
    plt.plot(np.arange(len(loss_seq)), loss_seq, 'ro-', linewidth=5)
    plt.axis([0, MAX_EPOCH, 0, 25])
    plt.savefig(os.path.join(IMG_DIR, "loss_curve_ep_{}.png".format(epoch)),
                format="PNG")
    plt.close()


def train_one_epoch(epoch, sess, ops, loss_seq):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    lr_list = [0.002, 0.0002, 0.00005, 0.001, 0.001, 0.0005]
    lr = lr_list[epoch // 20]

    train, _, max_node = smiles_loader.load()

    current_data = train['X']
    current_label = train['y']
    laplacian = train['L']
    size_x = train['size']
    adj_matrix = train['adj_list']
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    pred_all = []
    loss_sum = 0
    updated_laps = []
    res_laps = []
    used_size_x = []
    used_adj_list = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['laplacian_pl']: laplacian[start_idx:end_idx],
                     ops['size_x_pl']: size_x[start_idx:end_idx],
                     ops['lr']: lr,
                     }
        step, _, loss_val, pred_val, updated_lap, res_lap = sess.run(
            [
                ops['step'],
                ops['train_op'],
                ops['loss'],
                ops['pred'],
                ops['laplacian'],
                ops['res_lap'],
            ],
            feed_dict=feed_dict)
        pred_all.append(pred_val)

        loss_sum += loss_val
        updated_laps.append(updated_lap)
        res_laps.append(res_lap)
        used_size_x.append(size_x[start_idx:end_idx])
        used_adj_list += adj_matrix[start_idx:end_idx]

    updated_laps = np.vstack(updated_laps)
    res_laps = np.vstack(res_laps)
    used_size_x = np.squeeze(np.concatenate(used_size_x, axis=0)).astype(np.int32)

    "check and draw residual graphs"
    # view_graph(epoch, updated_laps, res_laps, used_size_x, used_adj_list, loss_seq)

    loss_seq += [loss_sum / float(num_batches)]
    log_string('mean loss at epoch %s : %f' % (epoch, loss_sum / float(num_batches)))

    pred_all = np.squeeze(np.hstack(pred_all))
    gt_all = current_label[:end_idx]
    rms = sqrt(mean_squared_error(pred_all, gt_all))
    log_string('rmse at epoch %s: %f \n\n' % (epoch, rms))

    return loss_seq


def evaluate_one_epoch(epoch, sess, ops):
    is_training = False
    _, test, max_node = smiles_loader.load()

    current_data = test['X']
    current_label = test['y']
    laplacian = test['L']
    size_x = test['size']
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    pred_all = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx],
                     ops['laplacian_pl']: laplacian[start_idx:end_idx],
                     ops['size_x_pl']: size_x[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     }
        pred_val = sess.run(
            [
                ops['pred'],
            ],
            feed_dict=feed_dict)
        pred_all.append(pred_val)

    pred_all = np.squeeze(np.hstack(pred_all))
    gt_all = current_label[:end_idx]
    rms = sqrt(mean_squared_error(pred_all, gt_all))
    log_string('Testing: rmse at epoch %s: %f \n\n' % (epoch, rms))


def view_graph(epoch, graph_lap, res_laps, size_x, used_adj_list, loss_seq, sel_idx=6):
    import matplotlib.pyplot as plt

    assert graph_lap.shape[0] == res_laps.shape[0] == size_x.shape[0]
    adj_list = used_adj_list[sel_idx]

    # form edges
    edges = []
    for u, adj in enumerate(adj_list):
        edges += [(u, v) for v in adj]

    n_nodes = size_x[sel_idx]
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)

    for e in edges:
        G.add_edge(e[0], e[1], color='r', weight=4)

    plt.figure(0)
    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw(G, pos, edges=edges, edge_color=colors, width=weights)

    plt.tight_layout()
    plt.axis('on')
    plt.savefig(os.path.join(IMG_DIR,
                             "orig_graph_ep_{}.png".format(epoch)),
                format="PNG")
    # plt.close()

    res_graph = np.squeeze(res_laps[sel_idx])
    res_graph = res_graph[:n_nodes, :n_nodes]
    np.fill_diagonal(res_graph, 0.0)
    res_graph = np.abs(res_graph)

    num_keep = int(n_nodes * (n_nodes - 1) * 0.15)
    threshold = np.sort(res_graph, axis=None)[-num_keep]
    res_graph[res_graph < threshold] = 0.0
    res_graph[res_graph >= threshold] = 1.0

    # res_G = nx.Graph()
    # for i in range(n_nodes):
    #     res_G.add_node(i, color='red')

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if res_graph[i][j] > 0:
                G.add_edge(i, j, color='b', weight=2)

    # sparse_graph = scipy.sparse.csr_matrix(res_graph)
    # G2 = nx.from_scipy_sparse_matrix(sparse_graph)
    # pos = nx.spring_layout(G2)
    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    weights = [G[u][v]['weight'] for u, v in edges]

    pos = nx.circular_layout(G)
    nx.draw(G, pos, edges=edges, edge_color=colors, width=weights)

    plt.tight_layout()
    plt.axis('on')
    plt.savefig(os.path.join(IMG_DIR,
                             "res_graph_ep_{}.png".format(epoch)),
                format="PNG")
    plt.close()

    display_heatmap(np.squeeze(res_laps[sel_idx]), n_nodes, epoch)

    plot_loss(epoch, loss_seq)


def display_heatmap(img, n_nodes, epoch):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # img = np.squeeze(img)
    img = img[:n_nodes, :n_nodes]
    img *= (255.0 / img.max())

    # img = img.astype(np.int32)
    # img = np.abs(img)

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(img, interpolation='nearest', cmap='hot')

    cbar = fig.colorbar(imgplot, ticks=[0, 255])
    cbar.ax.set_yticklabels(['< 0', '> 255'])

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    fig.savefig(os.path.join(IMG_DIR,
                             "res_graph_heatmap_ep_{}.png".format(epoch)))
    plt.close()


def test_graph():
    G = nx.Graph()
    for i in range(10):
        G.add_node(i)

    for e in [(0,1), (3,4), (2,3)]:
        G.add_edge(e[0], e[1])
        G[e[0]][e[1]]['color']='blue'

    nx.draw(G)
    nx.draw(G, pos=nx.spectral_layout(G))


if __name__ == "__main__":
    train()

    # test_graph()