import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import csv
import networkx as nx
from sklearn.metrics import roc_curve, roc_auc_score


BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import WSI_TCGA_loader
import smiles_loader
import io_utils


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='AGCN',
                    help='Model name: AGCN[default: AGCN]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1000, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=82, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=2e4, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

# MODLE_NAME = 'chebyshev'     # 'chebyshev'
BATCH_SIZE = FLAGS.batch_size
MAX_NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

DATASET_NAME = 'processed_{}maxnode'.format(MAX_NUM_POINT)
NUM_FEATURE = 128
NUM_CLASSES = 2

EVAl_FREQ = 5


MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

RESULT_DIR = os.path.join(LOG_DIR, 'wsi_tcga_results')
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

IMG_DIR = os.path.join(LOG_DIR, 'wsi_tcga_roc')
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

CSV_DIR = os.path.join(LOG_DIR, 'wsi_tcga_csv')
if not os.path.exists(CSV_DIR):
    os.mkdir(CSV_DIR)

CSV_DIR2 = os.path.join(LOG_DIR, 'wsi_tcga_csv_train')
if not os.path.exists(CSV_DIR2):
    os.mkdir(CSV_DIR2)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_wsi.txt'), 'w')
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
    model_list = ['agcn', 'chebyshev']
    for model_name in model_list:
        print("working on model %s" % model_name)

        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(GPU_INDEX)):
                data_pl, labels_pl, laplacian_pl, size_x_pl = MODEL.placeholder_inputs(BATCH_SIZE,
                                                                                       MAX_NUM_POINT,
                                                                                       NUM_FEATURE)
                is_training_pl = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                if model_name == 'agcn':
                    pred, end_points = MODEL.basic_agcn(data_pl,
                                                        laplacian_pl,
                                                        size_x_pl,
                                                        MAX_NUM_POINT,
                                                        NUM_CLASSES,
                                                        is_training_pl,
                                                        bn_decay=bn_decay)
                    loss = MODEL.agcn_loss(pred, end_points['regs_list'], labels_pl, reg_weight=1)

                elif model_name == 'chebyshev':
                    pred, end_points = MODEL.gcn_chebyshev(data_pl,
                                                           laplacian_pl,
                                                           size_x_pl,
                                                           MAX_NUM_POINT,
                                                           NUM_CLASSES)
                    loss = MODEL.cross_entropy(pred, labels_pl)
                else:
                    ValueError("No Such Model Name{}".format(model_name))

                # loss_reg = MODEL.loss_reg(end_points['regs_list'])

                seg_pred = MODEL.seg_pred(end_points['output_seg'])

                # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                # tf.summary.scalar('accuracy', accuracy)

                # Get training operator
                learning_rate = get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)

                optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

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
                   # 'laplacian': end_points['lap'],
                   # 'res_lap': end_points['res_lap'],
                   'pred': end_points['output_softmax'],
                   'seg_pred': seg_pred,
                   'loss': loss,
                   'train_op': train_op,
                   'step': batch}

            loss_seq = []
            accuracy_seq, measure_ep_seq = [], []

            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % epoch)
                sys.stdout.flush()

                loss_seq = train_one_epoch(epoch, sess, ops, loss_seq)

                if epoch % EVAl_FREQ == 0:
                    accuracy_seq = evaluate_one_epoch(epoch, sess, ops, accuracy_seq, model_name)
                    measure_ep_seq += [epoch]
            # plot_accuracy(measure_ep_seq, accuracy_seq)
            # plot_loss(MAX_EPOCH, loss_seq)


def plot_accuracy(measure_ep_seq, accuracy_seq):
    assert len(measure_ep_seq) == len(accuracy_seq)
    import matplotlib.pyplot as plt
    plt.figure(3)
    plt.plot(measure_ep_seq, accuracy_seq, 'ro-', linewidth=5)
    plt.axis([0, MAX_EPOCH, 0, 1])
    plt.savefig(os.path.join(IMG_DIR,
                             "accuracy_curves_ep_{}.png".format(MAX_EPOCH)),
                format="PNG")
    plt.close()


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

    train, _, max_node = WSI_TCGA_loader.load(DATASET_NAME)
    # train, _, max_node = smiles_loader.load()

    print("Data Loaded! \n")

    current_data = train['X']
    current_label = train['y']
    laplacian = train['L']
    size_x = train['size']
    wsi_name = train['name']
    patch_img_names = train['node_img']
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_seen = 0
    total_correct = 0
    loss_sum = 0
    updated_laps = []
    res_laps = []
    used_size_x = []
    all_class_pred, all_seg_pred = [], []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['laplacian_pl']: laplacian[start_idx:end_idx],
                     ops['size_x_pl']: size_x[start_idx:end_idx],
                     }
        step, _, loss_val, seg_pred, pred_val = sess.run(
            [
                ops['step'],
                ops['train_op'],
                ops['loss'],
                ops['seg_pred'],
                ops['pred'],
                # ops['laplacian'],
                # ops['res_lap'],
            ],
            feed_dict=feed_dict)

        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        # updated_laps.append(updated_lap)
        # res_laps.append(res_lap)
        used_size_x.append(size_x[start_idx:end_idx])

        all_class_pred.append(pred_val)
        all_seg_pred.append(seg_pred)

    # updated_laps = np.vstack(updated_laps)
    # res_laps = np.vstack(res_laps)
    used_size_x = np.squeeze(np.concatenate(used_size_x, axis=0)).astype(np.int32)
    all_class_pred = np.squeeze(np.concatenate(all_class_pred))
    all_seg_pred = np.squeeze(np.vstack(all_seg_pred))
    loss_seq += [loss_sum / float(num_batches)]

    # if epoch < 15:
    # view_graph(epoch, updated_laps, res_laps, used_size_x, used_adj_list, loss_seq)

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))

    # if epoch == MAX_EPOCH - EVAl_FREQ:
    #     # last validation, save node prediction
    #     save_highscore_node(all_seg_pred, size_x[:end_idx], all_class_pred,
    #                         wsi_name[:end_idx], patch_img_names[:end_idx], CSV_DIR2)

    return loss_seq


def print_seg(seg_pred, size_x, pred_class, wsi, patch):
    assert seg_pred.shape[0] == size_x.shape[0] == pred_class.shape[0]
    elm_idx = 0
    node_scores = seg_pred[elm_idx, 0: size_x[elm_idx], np.squeeze(pred_class[elm_idx])]
    sorted = np.argsort(np.squeeze(node_scores))  # patch id according to scores, small -> large

    print(seg_pred[elm_idx, 0: size_x[elm_idx], np.squeeze(pred_class[elm_idx])])
    print(wsi[elm_idx])
    print(patch[elm_idx][sorted[-1]])


def save_highscore_node(seg_pred, size_x, pred_class, wsi_name, patch_img_names, save_dir):
    # node ---> patch
    # seg_pred ---> node-wise prediction
    assert seg_pred.shape[0] == size_x.shape[0] == pred_class.shape[0]

    for wsi_idx in range(seg_pred.shape[0]):
        wsi = wsi_name[wsi_idx]
        patch_names = np.squeeze(patch_img_names[wsi_idx])
        "only output the predicted class's score (probs) at each nodes"
        node_scores = seg_pred[wsi_idx, 0: size_x[wsi_idx], np.squeeze(pred_class[wsi_idx])]
        sorted_id = np.argsort(np.squeeze(node_scores))    # patch id according to scores, small -> large

        # print("This WSI has patch:{}".format(sorted_id.shape[0]))

        node_n = sorted_id.shape[0]
        "first 20%, middle 20%, high 20%. node is patch"
        node_idxs = {
            'low_score': sorted_id[: int(node_n * 0.4)],
            'middle_score': sorted_id[int(node_n * 0.4): int(node_n * 0.8)],
            'high_score': sorted_id[int(node_n * 0.8):],
        }

        # category --> categorize the degree of dependency of patch on cancel class
        row_header = ['patch_id', 'category', 'patch_img', 'score']
        rows_low = [[n_id, 'low', patch_names[n_id], node_scores[n_id]] for n_id in node_idxs['low_score']]
        rows_middle = [[n_id, 'middle', patch_names[n_id], node_scores[n_id]] for n_id in node_idxs['middle_score']]
        rows_high = [[n_id, 'high', patch_names[n_id], node_scores[n_id]] for n_id in node_idxs['high_score']]

        all_rows = []
        all_rows += rows_low
        all_rows += rows_middle
        all_rows += rows_high
        with open(os.path.join(save_dir, 'selected_patch_wsi_{}.csv'.format(wsi[:-4])), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|',)
            writer.writerow(row_header)
            for r in all_rows:
                writer.writerow(r)


def evaluate_one_epoch(epoch, sess, ops, accuracy_seq, model_name):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    _, test, max_node = WSI_TCGA_loader.load(DATASET_NAME)
    # _, test, max_node = smiles_loader.load()

    print("Data Loaded! \n")

    current_data = test['X']
    current_label = test['y']
    laplacian = test['L']
    size_x = test['size']
    wsi_name = test['name']
    patch_img_names = test['node_img']
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    all_seg_pred = []
    all_class_pred = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['laplacian_pl']: laplacian[start_idx:end_idx],
                     ops['size_x_pl']: size_x[start_idx:end_idx],
                     }
        seg_pred, pred_val = sess.run(
            [
                # ops['step'],
                # ops['train_op'],
                # ops['loss'],
                ops['seg_pred'],
                ops['pred'],
            ],
            feed_dict=feed_dict)

        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE

        all_class_pred.append(pred_val)
        all_seg_pred.append(seg_pred)

        # print_seg(seg_pred, size_x[start_idx:end_idx], pred_val,
        #           wsi_name[start_idx:end_idx], patch_img_names[start_idx:end_idx])

    all_class_pred = np.squeeze(np.concatenate(all_class_pred))
    all_seg_pred = np.squeeze(np.vstack(all_seg_pred))

    # last validation, save node prediction
    # save_highscore_node(all_seg_pred, size_x[:end_idx], all_class_pred,
    #                     wsi_name[:end_idx], patch_img_names[:end_idx], CSV_DIR)

    save_result(epoch, all_class_pred, current_label[: end_idx], model_name)
    # plot_roc(all_class_pred, current_label[: end_idx])

    log_string('accuracy: %f' % (total_correct / float(total_seen)))

    accuracy_seq += [total_correct / float(total_seen)]
    return accuracy_seq


def save_result(epoch, preds, labels, model_name):
    """
    save the predictions to local
    """
    out_preds = "prediction_%s_ep_%d.joblib" % (model_name, epoch)
    print("saving prediction to {}".format(out_preds))
    io_utils.save_to_disk(preds, os.path.join(RESULT_DIR, out_preds))

    out_labels = "label_{}.joblib".format(model_name)
    if not os.path.exists(os.path.join(RESULT_DIR, out_labels)):
        print("saving prediction to {}".format(out_labels))
        io_utils.save_to_disk(labels, os.path.join(RESULT_DIR, out_labels))

    print("Successfully Saved Results!")


def load_and_plot_roc():
    model_list = ['agcn', 'chebyshev']
    all_pred, all_label = [], []
    for model in model_list:
        pred_file = 'prediction_{}_ep_40.joblib'.format(model)
        label_file = "label_{}.joblib".format(model)
        pred = np.array(io_utils.load_from_disk(os.path.join(RESULT_DIR, pred_file)))
        label = np.array(io_utils.load_from_disk(os.path.join(RESULT_DIR, label_file)))
        all_pred.append(pred)
        all_label.append(label)

    plot_roc(all_pred, all_label, model_list)


def plot_roc(pred_prob_list, gt_list, model_list):
    "plot more than one ROC curves in one figure"
    import matplotlib.pyplot as plt
    assert len(pred_prob_list) == len(gt_list) == len(model_list)

    plt.figure()
    colors = ['darkorange', 'red', 'cyan', 'magenta', 'blue']

    for l_idx, (pred_prob, gt, model) in enumerate(zip(pred_prob_list, gt_list, model_list)):
        assert pred_prob.shape[0] == gt.shape[0]

        fpr, tpr, threshold = roc_curve(gt, pred_prob, pos_label=1)
        auc = roc_auc_score(gt, pred_prob, 'samples')

        print("ROC-AUC score %f" % auc)

        plt.plot(fpr, tpr, color=colors[int(l_idx % len(colors))],
                 lw=2, label='%s : ROC curve (area = %0.2f)' % (model, auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()

    plt.savefig(os.path.join(IMG_DIR, "roc_auc_models_cmp.png"), format="PNG")


def view_graph(epoch, graph_lap, res_laps, size_x, used_adj_list, loss_seq, sel_idx=6):
    import matplotlib.pyplot as plt
    import scipy.sparse

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
    # load_and_plot_roc()
