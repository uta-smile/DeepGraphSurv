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
from lifelines.utils import concordance_index


BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import WSI_NLST_loader
import WSI_TCGA_loader
import prepare_data
import io_utils


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='AGCN',
                    help='Model name: AGCN[default: AGCN]')
parser.add_argument('--log_dir', default='log/agcn_survival_ADC', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=500, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=2000, help='Decay step for lr decay [default: 200000]')
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
MODEL_NAME = 'chebyshev'
DATA_NAME = 'NLST'

parameter_pack = {
    'lr': BASE_LEARNING_RATE,
    'max_node': MAX_NUM_POINT,
    'batch_size': BATCH_SIZE,
}

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')


BASE_LOG_DIR = 'log/%s_survival_%s' % (MODEL_NAME, DATA_NAME)
if not os.path.exists(BASE_LOG_DIR):
    os.mkdir(BASE_LOG_DIR)

LOG_DIR = os.path.join(BASE_LOG_DIR, 'node_%s_batch_%s' % (MAX_NUM_POINT, BATCH_SIZE))
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

RESULT_DIR = os.path.join(LOG_DIR, 'results')
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

FEATURE_DIR = os.path.join(LOG_DIR, 'features')
if not os.path.exists(FEATURE_DIR):
    os.mkdir(FEATURE_DIR)

IMG_DIR = os.path.join(LOG_DIR, 'image_save')
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

MODEL_SAVE_DIR = os.path.join(LOG_DIR, 'model_save')
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)

LABEL_DIR = os.path.join(BASE_DIR, 'models/survival_labels')
assert os.path.exists(LABEL_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_%s_survival_%s.txt' % (MODEL_NAME, DATASET_NAME)), 'w')
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


def get_learning_rate(batch, base_lr):
    learning_rate = tf.train.exponential_decay(
        base_lr,  # Base learning rate.
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
    # model_list = ['agcn']
    # for model_name in model_list:
    #     print("working on model %s" % model_name)

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            data_pl, labels_pl, laplacian_pl, size_x_pl = MODEL.placeholder_inputs_survival(
                                                                                   BATCH_SIZE,
                                                                                   MAX_NUM_POINT,
                                                                                   NUM_FEATURE)
            status_pl = MODEL.placeholder_survival(BATCH_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            learning_rate_pl = tf.placeholder(tf.float32, shape=())

            dc_learning_rate = get_learning_rate(batch, base_lr=learning_rate_pl)

            # Get model and loss
            if MODEL_NAME == 'agcn' or MODEL_NAME == 'agcn_attn':
                end_points = MODEL.basic_agcn_wsi_survival2(data_pl,
                                                           laplacian_pl,
                                                           size_x_pl,
                                                           MAX_NUM_POINT,
                                                           1,
                                                           is_training_pl,
                                                           use_attn=True,
                                                           bn_decay=bn_decay)
                loss = MODEL.survival_loss(end_points['output'], status_pl)

                optimizer = tf.train.AdamOptimizer(dc_learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

                # optimizer = tf.train.AdamOptimizer(learning_rate_pl)
                # train_op = optimizer.minimize(loss, global_step=batch)

            elif MODEL_NAME == 'chebyshev' or MODEL_NAME == 'chebyshev_attn':
                end_points = MODEL.gcn_chebyshev_survival(data_pl,
                                                          laplacian_pl,
                                                          size_x_pl,
                                                          MAX_NUM_POINT,
                                                          1,
                                                          is_training_pl,
                                                          use_attn=False,
                                                          bn_decay=None)
                loss = MODEL.survival_loss(end_points['output'], status_pl)

                optimizer = tf.train.AdamOptimizer(dc_learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)
            else:
                ValueError("No Such Model Name {}".format(MODEL_NAME))
                return
            # loss_reg = MODEL.loss_reg(end_points['regs_list'])

            # Get training operator
            # learning_rate = get_learning_rate(batch)
            # tf.summary.scalar('learning_rate', learning_rate)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        # init = tf.global_variables_initializer()

        ops = {'pointclouds_pl': data_pl,
               'status_pl': status_pl,
               'laplacian_pl': laplacian_pl,
               'size_x_pl': size_x_pl,
               # 'attn': end_points['attn'],
               'is_training_pl': is_training_pl,
               'final_lr': dc_learning_rate,
               'learning_rate_pl': learning_rate_pl,
               'pred': end_points['output'],
               'graph_feature': end_points['graph_feature'],
               'loss': loss,
               'train_op': train_op,
               'saver_op': end_points['saver'],
               'step': batch}

        # if os.path.exists(MODEL_SAVE_DIR + '/agcn_survival_backbone_ep50.meta'):
            # load model and evaluation
            # restore_eval(sess, ops)
        # else:
        # train model
        max_repeat = 1
        for trial_id, lr in enumerate([0.0002]):
            log_string("processing %s trial on learning rate %f, batch size %s" % (trial_id, lr, BATCH_SIZE))

            parameter_pack = {
                'lr': lr,
                'max_node': MAX_NUM_POINT,
                'batch_size': BATCH_SIZE,
            }
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init,
                     {is_training_pl: True})

            cross_validation = {
                'avg_ci': [],
                'oneshot_ci': [],
                'patient_ci': [],
            }
            loss_seq = [0.0] * MAX_EPOCH
            train_ci = [0.0] * MAX_EPOCH
            for _ in range(max_repeat):
                p_value_list, c_index_list = [], []
                patient_ci_list, oneshot_ci_list = [], []
                for epoch in range(MAX_EPOCH):
                    # log_string('**** EPOCH %03d ****' % epoch)
                    sys.stdout.flush()

                    train_data_dict, loss = train_one_epoch(epoch, sess, ops, lr)
                    loss_seq[epoch] += loss
                    train_ci[epoch] += train_data_dict['train_ci']
                    if epoch % 5 == 0:
                        save_graph_feature(train_data_dict, epoch)

                    if epoch > 0 and epoch % 5 == 0:
                        test_data_dict = evaluate_one_epoch(epoch, sess, ops)

                        c_index_list += [test_data_dict['ci']]
                        oneshot_ci_list += [test_data_dict['oneshot_ci']]
                        patient_ci_list += [test_data_dict['p_ci']]
                        p_value_list += [test_data_dict['p_value']]
                        save_graph_feature(test_data_dict, epoch, istrain=False)

                """ train on LASSO COX using the latest predictions"""
                # train_cox_lasso(train_data_dict, test_data_dict)
                # save_feature_csv(train_data_dict, test_data_dict)

                best_set_ci, best_set_os_ci, best_set_p_ci = statistic_results(c_index_list,
                                                                               oneshot_ci_list,
                                                                               patient_ci_list,
                                                                               p_value_list)

                cross_validation['avg_ci'].append(best_set_ci)
                cross_validation['oneshot_ci'].append(best_set_os_ci)
                cross_validation['patient_ci'].append(best_set_p_ci)

            # calculate , print CI after cross-validation
            process_cv(cross_validation, parameter_pack, trial_id)
            # cal and save the  (avg.) loss sequence for model comparison
            save_loss_seq([l / float(MAX_EPOCH) for l in loss_seq], parameter_pack)
            save_ci_seq([l / float(MAX_EPOCH) for l in train_ci], parameter_pack)


def process_cv(cv, p_set, idx):

    lr = p_set['lr']
    batch_size = p_set['batch_size']

    list_ci, list_p_value = [], []
    for s in cv['avg_ci']:
        list_ci += [s['ci']]
        list_p_value += [s['p_value']]
    list_ci = np.array(list_ci)
    # list_p_value = np.array(list_p_value)
    log_string('min CI: %f, max CI: %f' % (np.min(list_ci), np.max(list_ci)))
    log_string('avg. CI: %f, CI std.: %f\n' % (float(np.mean(list_ci)), float(np.std(list_ci))))

    list_os_ci, list_p_value = [], []
    for s in cv['oneshot_ci']:
        list_os_ci += [s['ci']]
        list_p_value += [s['p_value']]
    list_os_ci = np.array(list_os_ci)
    # list_p_value = np.array(list_p_value)
    log_string('min oneshot CI: %f, max oneshot CI: %f' % (np.min(list_os_ci), np.max(list_os_ci)))
    log_string('avg. oneshot CI: %f, oneshot CI std.: %f\n' % (float(np.mean(list_os_ci)), float(np.std(list_os_ci))))

    list_p_ci, list_p_value = [], []
    for s in cv['patient_ci']:
        list_p_ci += [s['ci']]
        list_p_value += [s['p_value']]
    list_p_ci = np.array(list_p_ci)
    # list_p_value = np.array(list_p_value)
    log_string('min patient CI: %f, max patient CI: %f' % (np.min(list_p_ci), np.max(list_p_ci)))
    log_string('avg. patient CI: %f, patient CI std.: %f\n' % (float(np.mean(list_p_ci)), float(np.std(list_p_ci))))

    with open(os.path.join(RESULT_DIR, 'cv_%s_%s.csv' % (MODEL_NAME, DATA_NAME)), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if idx == 0:
            writer.writerow(['learning_rate',
                             'batch_size',
                             'max CI',
                             'avg CI',
                             'CI std',
                             'max oneshot CI',
                             'avg oneshot CI',
                             'oneshot CI std',
                             'max patient CI',
                             'avg patient CI',
                             'patient CI std'])

        writer.writerow([lr] + [batch_size] + [np.max(list_ci)] + [float(np.mean(list_ci))] + [float(np.std(list_ci))]
                        + [np.max(list_os_ci)] + [np.mean(list_os_ci)] + [np.std(list_os_ci)]
                        + [np.max(list_p_ci)] + [np.mean(list_p_ci)] + [np.std(list_p_ci)])


def statistic_results(ci_list, oneshot_ci_list, patient_ci_list, p_list):
    """ from the training process, select best results (early stopping) and save corres. p_value and epoch"""
    best_ci = float(np.max(np.array(ci_list)))
    idx = np.argmax(np.array(ci_list))
    best_p = float(p_list[idx])
    best_epoch = (idx + 1) * 5
    best_avg_ci = {
        'p_value': best_p,
        'ci': best_ci,
        'epoch': best_epoch
    }

    best_os_ci = float(np.max(np.array(oneshot_ci_list)))
    idx = np.argmax(np.array(oneshot_ci_list))
    best_p = float(p_list[idx])
    best_epoch_os = (idx + 1) * 5
    best_oneshot_ci = {
        'p_value': best_p,
        'ci': best_os_ci,
        'epoch': best_epoch_os
    }

    best_p_ci = float(np.max(np.array(patient_ci_list)))
    idx = np.argmax(np.array(patient_ci_list))
    best_p = float(p_list[idx])
    best_epoch_p = (idx + 1) * 5
    best_patient_ci = {
        'p_value': best_p,
        'ci': best_p_ci,
        'epoch': best_epoch_p
    }
    return best_avg_ci, best_oneshot_ci, best_patient_ci


def save_loss_seq(loss_seq, p_set):
    import csv
    lr = p_set['lr']
    batch_size = p_set['batch_size']

    assert type(loss_seq) == list
    with open(os.path.join(RESULT_DIR, 'loss_%s_%s.csv' % (MODEL_NAME, DATA_NAME)), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['learning_rate', 'batch_size'])
        writer.writerow([lr] + [batch_size])
        writer.writerow(['losses sequence'])
        writer.writerow(loss_seq)


def save_ci_seq(ci_seq, p_set):
    import csv
    lr = p_set['lr']
    batch_size = p_set['batch_size']

    assert type(ci_seq) == list
    with open(os.path.join(RESULT_DIR, 'c_index_%s_%s.csv' % (MODEL_NAME, DATA_NAME)), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['learning_rate', 'batch_size'])
        writer.writerow([lr] + [batch_size])
        writer.writerow(['ci sequence'])
        writer.writerow(ci_seq)


def save_graph_feature(result_dict, epoch, istrain=True):
    features = result_dict['patient_feature']
    status = result_dict['status']
    time = result_dict['time']
    pid = result_dict['pid']

    if istrain:
        csv_file = 'train_feature_%s_%s_ep%s.csv' % (MODEL_NAME, DATA_NAME, epoch)
    else:
        csv_file = 'test_feature_%s_%s_ep%s.csv' % (MODEL_NAME, DATA_NAME, epoch)

    with open(os.path.join(FEATURE_DIR, csv_file), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['pid', 'status', 'time', 'features'])
        for idx, p in enumerate(np.unique(pid).tolist()):
            p_idx = np.where(pid == p)
            p_feature = features[p]
            p_status = status[p_idx]
            p_time = time[p_idx]
            writer.writerow([p] + [p_status[0].tolist()] + [p_time[0].tolist()] + p_feature.tolist())


# def save_feature_csv(train, test, epoch):
#     import csv
#
#     train_graph_feature = train['features']
#     train_status = train['status']
#     train_time = train['time']
#     train_pid = train['pid']
#
#     with open(os.path.join(FEATURE_DIR, 'train_%s_%s_%d.csv' % (MODEL_NAME, DATA_NAME, epoch)), 'w') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#
#         writer.writerow(['pid', 'status', 'survival_time', 'feature'])
#         for idx, row in enumerate(train_graph_feature.tolist()):
#             writer.writerow([train_pid[idx]] + [train_status[idx]] + [train_time[idx]] + row)
#
#     test_graph_feature = test['features']
#     test_status = test['status']
#     test_time = test['time']
#     test_pid = test['pid']
#
#     with open(os.path.join(FEATURE_DIR, 'test_%s_%s_%d.csv' % (MODEL_NAME, DATA_NAME, epoch)), 'w') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#
#         writer.writerow(['pid', 'status', 'survival_time', 'feature'])
#         for idx, row in enumerate(test_graph_feature.tolist()):
#             writer.writerow([test_pid[idx]] + [test_status[idx]] + [test_time[idx]] + row)


def retrieve_feature(sess, ops, data, laplacian, size_x, time, status, test_pid, is_training=False):

    """
    Parse data to network, get predicted survival risk, graph features(WSI wise),
     and calculate and print CI on testing
    :param sess:
    :param ops:
    :param data:
    :param laplacian:
    :param size_x:
    :param time:
    :param status:
    :param is_training:
    :return:
    """
    file_size = data.shape[0]
    num_batches = file_size // BATCH_SIZE
    test_ci = 0.0

    if num_batches == 0:
        # data size < BATCH_SIZE
        tile_times = int(BATCH_SIZE // file_size) + 1
        data_tiled = np.tile(data, (tile_times, 1, 1))[: BATCH_SIZE]
        laplacian_tiled = np.tile(laplacian, (tile_times, 1, 1))[: BATCH_SIZE]
        size_x_tiled = np.tile(size_x, reps=tile_times)[: BATCH_SIZE]
        feed_dict = {ops['pointclouds_pl']: data_tiled,
                     ops['is_training_pl']: is_training,
                     ops['laplacian_pl']: laplacian_tiled,
                     ops['size_x_pl']: size_x_tiled,
                     }
        pred_val, graph_feature = sess.run(
            [
                ops['pred'],
                ops['graph_feature']
            ],
            feed_dict=feed_dict)
        all_preds = -np.exp(pred_val[: file_size].ravel())  # pred_val -> predicted survival time
        all_features = graph_feature[: file_size]
        test_ci += concordance_index(time, all_preds, status)
        log_string('Testing CI : %f \n\n' % test_ci)
        avg_test_ci = test_ci
        test_ci_oneshot = test_ci
        patient_ci = get_patient_ci(time[: file_size],
                                    all_preds,
                                    status[: file_size],
                                    test_pid[: file_size])

    else:
        batch_count = 0.0
        all_features, all_preds, all_status, all_times = [], [], [], []
        for batch_idx in range(num_batches):
            batch_count += 1
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: data[start_idx:end_idx],
                         ops['is_training_pl']: is_training,
                         ops['laplacian_pl']: laplacian[start_idx:end_idx],
                         ops['size_x_pl']: size_x[start_idx:end_idx],
                         }
            pred_val, graph_feature = sess.run(
                [
                    ops['pred'],
                    ops['graph_feature']
                ],
                feed_dict=feed_dict)
            all_preds.append(-np.exp(pred_val.ravel()))  # pred_val -> predicted survival time
            all_features.append(graph_feature)
            test_ci += concordance_index(time[start_idx:end_idx],
                                         -np.exp(pred_val.ravel()),
                                         status[start_idx:end_idx])
            all_status.append(status[start_idx:end_idx])
            all_times.append(time[start_idx:end_idx])

        left_num = file_size - end_idx
        pad_num = BATCH_SIZE - left_num
        if pad_num > 0 and left_num > 0:
            batch_count += 1
            last_batch_data = np.concatenate((data[-left_num:], data[: pad_num]), axis=0)
            last_batch_lap = np.concatenate((laplacian[-left_num:], laplacian[: pad_num]), axis=0)
            last_batch_size_x = np.concatenate((size_x[-left_num:], size_x[: pad_num]), axis=0)

            feed_dict = {ops['pointclouds_pl']: last_batch_data,
                         ops['is_training_pl']: is_training,
                         ops['laplacian_pl']: last_batch_lap,
                         ops['size_x_pl']: last_batch_size_x,
                         }
            pred_val, graph_feature = sess.run(
                [
                    ops['pred'],
                    ops['graph_feature']
                ],
                feed_dict=feed_dict)
            all_preds.append(-np.exp(pred_val[: left_num].ravel()))  # pred_val -> predicted survival time
            all_features.append(graph_feature[: left_num])
            test_ci += concordance_index(time[-left_num:],
                                         -np.exp(pred_val[: left_num].ravel()),
                                         status[-left_num:])
            all_status.append(status[-left_num:])
            all_times.append(time[-left_num:])

        all_preds = np.squeeze(np.concatenate(all_preds))  #
        all_features = np.squeeze(np.concatenate(all_features))
        log_string('Testing CI : %f \n\n' % (test_ci / float(batch_count)))
        avg_test_ci = test_ci / float(batch_count)

        all_status = np.squeeze(np.concatenate(all_status))
        all_times = np.squeeze(np.concatenate(all_times))
        test_ci_oneshot = concordance_index(all_times, all_preds, all_status)
        log_string('One-shot Testing CI : %f \n\n' % test_ci_oneshot)

        patient_ci = get_patient_ci(all_times, all_preds, all_status, test_pid)

        p_feature = get_patient_feature(test_pid, all_features)

    assert all_features.shape[0] == data.shape[0] == all_preds.shape[0]
    return all_features, all_preds, avg_test_ci, test_ci_oneshot, patient_ci, p_feature


def get_patient_ci(times, preds, status, pids):
    # due the DNN give WSI-wise prediction, some patient may have multiple prediction of survival times
    # we have several ways of doing it, we choose to averaging the prediction and use it as pid wise prediction
    all_patient = np.unique(pids).tolist()
    all_p_preds, all_p_times, all_p_status = [], [], []
    for p in all_patient:
        idxs = np.where(pids == p)
        preds_p = preds[idxs]
        times_p = times[idxs]
        status_p = status[idxs]
        avg_pred_p = np.squeeze(np.mean(preds_p))
        avg_times_p = times_p[0]
        avg_status_p = status_p[0]
        all_p_preds.append(avg_pred_p)
        all_p_times.append(avg_times_p)
        all_p_status.append(avg_status_p)
    all_p_preds = np.array(all_p_preds)
    all_p_times = np.array(all_p_times)
    all_p_status = np.array(all_p_status)

    patient_ci = concordance_index(all_p_times, all_p_preds, all_p_status)
    log_string('Patient Testing CI : %f \n\n' % patient_ci)
    return patient_ci


# def train_cox_lasso(train_data, test_data, l1=0.5, alphas=np.linspace(0.001, 0.5, 40)):
#
#     from sksurv.linear_model import CoxnetSurvivalAnalysis
#     cph = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=alphas)
#     train_y = []
#
#     # prepare data, train Lasso Cox model
#     features = train_data['features']
#     status = train_data['status'].tolist()
#     time = train_data['time'].tolist()
#
#     for i in range(features.shape[0]):
#         train_y.append((status[i], time[i]))
#
#     train_y = np.array(train_y, dtype=[('Status', '?'), ('Survival', '<f8')])
#     cph.fit(features, train_y)
#
#     # predict survival time on testing
#     train_res = cph.predict(features)
#     train_ci = concordance_index(np.array(time), train_res, np.array(status))
#     print("LASSO COX Train CI %f" % train_ci)
#     res_median = np.median(train_res)
#
#     features_test = test_data['features']
#     test_res = cph.predict(features_test)
#     test_status = test_data['status'].tolist()
#     test_time = test_data['time'].tolist()
#
#     test_ci = concordance_index(np.array(test_time), test_res, np.array(test_status))
#     print("LASSO COX Testing CI %f" % test_ci)
#     group1 = test_res > res_median

    # if sum(group1) > 0:
    #     draw_curve(test_res, test_status, test_time, median_fromtrain)
    # else:
    #     ValueError('group1 is empty! ')


def plot_loss(epoch, loss_seq):
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.plot(np.arange(len(loss_seq)), loss_seq, 'ro-', linewidth=5)
    plt.axis([0, MAX_EPOCH, 0, 25])
    plt.savefig(os.path.join(IMG_DIR, "loss_curve_ep_{}.png".format(epoch)),
                format="PNG")
    plt.close()


def find_labels(wsi_names, labels, labels2=None, dataset=DATA_NAME):

    wsi_status_list, wsi_time_list, wsi_pid_list = [], [], []
    if dataset == 'TCGA' or dataset == 'TCGA-LUAD' or dataset == 'TCGA-LUSC' or dataset == 'TCGA-GBM':
        for wsi in wsi_names.tolist():
            wsi_id = wsi[:12]
            # print("finding labels for WSI %s" % wsi_id)

            if wsi_id in labels['pid']:
                # here wsi_id is pid
                idx = labels['pid'].index(wsi_id)
                status = float(labels['status'][idx])
                time = float(labels['time'][idx])
                pid = wsi_id
                wsi_status_list.append(status)
                wsi_time_list.append(time)
                wsi_pid_list.append(pid)
            else:
                ValueError("This WSI %s has no label" % wsi_id)

    elif dataset == 'NLST':
        for wsi in wsi_names.tolist():
            wsi_id = wsi[:-4] + '.svs'
            # print("finding labels for WSI %s" % wsi_id)

            if wsi_id in labels['wsi']:
                idx = labels['wsi'].index(wsi_id)
                status = float(labels['status'][idx])
                time = float(labels['time'][idx])
                pid = labels['pid'][idx]
                wsi_status_list.append(status)
                wsi_time_list.append(time)
                wsi_pid_list.append(pid)
            else:
                ValueError("This WSI %s has no label" % wsi_id)

    elif dataset == 'ADC' or dataset == 'SCC':
        for wsi in wsi_names.tolist():

            # do not know if the wsi from TCGA or NLST
            wsi_id = wsi[:-4] + '.svs'  # try NLST

            if wsi_id in labels['wsi']:
                idx = labels['wsi'].index(wsi_id)
                status = float(labels['status'][idx])
                time = float(labels['time'][idx])
                pid = labels['pid'][idx]
                wsi_status_list.append(status)
                wsi_time_list.append(time)
                wsi_pid_list.append(pid)
                continue

            # try TCGA
            wsi_id = wsi[:12]
            if wsi_id in labels2['pid']:
                # here wsi_id is pid
                idx = labels2['pid'].index(wsi_id)
                status = float(labels2['status'][idx])
                time = float(labels2['time'][idx])
                pid = wsi_id
                wsi_status_list.append(status)
                wsi_time_list.append(time)
                wsi_pid_list.append(pid)

            else:
                ValueError("This WSI %s has no label" % wsi_id)

    return np.array(wsi_status_list).astype(np.float32), np.array(wsi_time_list).astype(np.float32), np.array(wsi_pid_list)


def train_one_epoch(epoch, sess, ops, lr):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    if epoch == 30:
        print('stop me!')

    data, laplacian, size_x, wsi_name = prepare_data.prepare_train_data(DATA_NAME, DATASET_NAME)

    file_size = data.shape[0]
    num_batches = file_size // BATCH_SIZE

    if DATA_NAME == 'ADC' or DATA_NAME == 'SCC':
        nlst_labels, tcga_labels = prepare_data.retrieve_survival_labels_train(DATA_NAME, DATASET_NAME, LABEL_DIR)
    else:
        labels = prepare_data.retrieve_survival_labels_train(DATA_NAME, DATASET_NAME, LABEL_DIR)

    loss_sum = 0
    train_ci = 0

    start_idx = 0
    end_idx = BATCH_SIZE

    all_features, all_status, all_time, all_pid, all_wsi_name = [], [], [], [], []

    # for batch_idx in range(num_batches):
    while end_idx <= file_size:
        wsi_sel = wsi_name[start_idx: end_idx]
        if DATA_NAME == 'ADC' or DATA_NAME == 'SCC':
            status, time, pid = find_labels(wsi_sel, nlst_labels, tcga_labels, DATA_NAME)
        else:
            status, time, pid = find_labels(wsi_sel, labels, dataset=DATA_NAME)

        if 1 not in status.tolist() or 0 not in status.tolist():
            # this batch cannot be run, because loss will be 0. extend the end_idx
            start_idx += 1
            end_idx = start_idx + BATCH_SIZE
        else:
            idx_order = np.argsort(time)[::-1]  # from larger to smaller
            batch_data = data[start_idx:end_idx]
            batch_lap = laplacian[start_idx:end_idx]
            batch_size_x = size_x[start_idx:end_idx]

            batch_data = batch_data[idx_order]
            batch_lap = batch_lap[idx_order]
            batch_size_x = batch_size_x[idx_order]
            status = status[idx_order]
            time = time[idx_order]
            pid = pid[idx_order]
            wsi_sel = wsi_sel[idx_order]
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['is_training_pl']: is_training,
                         ops['learning_rate_pl']: lr,
                         ops['laplacian_pl']: batch_lap,
                         ops['size_x_pl']: batch_size_x,
                         ops['status_pl']: status,
                         }
            step, _, real_time_lr, loss_val, pred_val, graph_feature = sess.run(
                [
                    ops['step'],
                    ops['train_op'],
                    ops['final_lr'],
                    ops['loss'],
                    ops['pred'],
                    # ops['attn'],
                    ops['graph_feature'],
                ],
                feed_dict=feed_dict)
            log_string(" lr at epoch % s : %s" % (epoch, real_time_lr))
            loss_sum += loss_val
            train_ci += concordance_index(time, -np.exp(pred_val.ravel()), status)
            start_idx = end_idx
            end_idx += BATCH_SIZE

            all_features.append(graph_feature)
            all_status.append(status)
            all_time.append(time)
            all_pid.append(pid)
            all_wsi_name.append(wsi_sel)

    all_features = np.squeeze(np.vstack(all_features))
    all_status = np.squeeze(np.concatenate(all_status))
    all_time = np.squeeze(np.concatenate(all_time))
    all_pid = np.squeeze(np.concatenate(all_pid))
    all_wsi_name = np.squeeze(np.concatenate(all_wsi_name))

    avg_loss = loss_sum / float(num_batches)
    log_string('Training CI at epoch %s: %f' % (epoch, train_ci / float(num_batches)))
    log_string('mean loss at epoch %s: %f \n\n' % (epoch, avg_loss))

    if epoch > 19 and epoch % 10 == 0:
        saver = ops['saver_op']
        save_path = saver.save(sess, MODEL_SAVE_DIR + '/agcn_survival_backbone_ep{}'.format(epoch))
        print("Backbone Net saved in file: %s" % save_path)

        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

        print_tensors_in_checkpoint_file(file_name=save_path,
                                         tensor_name='',
                                         all_tensors=False)

    p_feature = get_patient_feature(all_pid, all_features)

    train_data_dict = {
        'features': all_features,
        'status': all_status,
        'time': all_time,
        'pid': all_pid,
        'wsi': all_wsi_name,
        'patient_feature': p_feature,
        'train_ci': train_ci / float(num_batches),
    }

    return train_data_dict, avg_loss


def get_patient_feature(pid, features):
    all_p_feature = {}
    for p in np.unique(pid).tolist():
        idx_wsi = np.where(pid == p)
        feature_p = features[idx_wsi]
        mean_f_p = np.mean(feature_p, axis=0)
        all_p_feature[p] = mean_f_p
    return all_p_feature


def evaluate_one_epoch(epoch, sess, ops):
    """ ops: dict mapping from string to tf ops """

    data, laplacian, size_x, wsi_name = prepare_data.prepare_test_data(DATA_NAME, DATASET_NAME)

    if DATA_NAME == 'ADC' or DATA_NAME == 'SCC':
        nlst_labels, tcga_labels = prepare_data.retrieve_survival_labels_test(DATA_NAME, DATASET_NAME, LABEL_DIR)
    else:
        labels = prepare_data.retrieve_survival_labels_test(DATA_NAME, DATASET_NAME, LABEL_DIR)

    if DATA_NAME == 'ADC' or DATA_NAME == 'SCC':
        test_status, test_time, test_pid = find_labels(wsi_name, nlst_labels, tcga_labels, DATA_NAME)
    else:
        test_status, test_time, test_pid = find_labels(wsi_name, labels, dataset=DATA_NAME)

    print("Evaluating on Testing")
    test_graph_features, test_pred_time, test_ci, oneshot_ci, patient_ci, p_feature = retrieve_feature(sess, ops,
                                                                    data,
                                                                    laplacian,
                                                                    size_x,
                                                                    test_time,
                                                                    test_status,
                                                                    test_pid)

    # eval all
    p_value = draw_curve(test_pred_time, test_status, test_time, np.median(test_pred_time), epoch)

    test_data_dict = {
        'features': test_graph_features,
        'status': test_status,
        'time': test_time,
        'pid': test_pid,
        'patient_feature': p_feature,
        'p_value': p_value,
        'ci': test_ci,
        'oneshot_ci': oneshot_ci,
        'p_ci': patient_ci,
    }
    return test_data_dict


def draw_curve(preds, status, time, median_fromtrain, epoch, save_fig=False):
    # Kaplan-Meier curve on testing data, group by median from train predicted survival time
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    T = np.array(time).astype(np.int32)
    E = np.array(status).astype(np.int32)

    ix = preds > median_fromtrain  # low risk group, preds ~ predicted survival time
    # kmf.fit(T[~ix], E[~ix], label='high risk')
    # ax = kmf.plot()
    #
    # kmf.fit(T[ix], E[ix], label='low risk')
    # kmf.plot(ax=ax)
    # if save_fig:
    #     ax.get_figure().savefig(os.path.join(IMG_DIR, "km_%s_%s_ep%d.png" % (MODEL_NAME, DATA_NAME, epoch)))

    from lifelines.statistics import logrank_test
    results = logrank_test(T[~ix], T[ix], event_observed_A=E[~ix], event_observed_B=E[ix])
    results.print_summary()
    print(results.p_value)
    print(results.test_statistic)

    return results.p_value


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


# def count_pid(data):


if __name__ == "__main__":
    train()

    # train_data, laplacian, size_x, wsi_name_train = prepare_data.prepare_train_data(DATA_NAME, DATASET_NAME)
    # test_data, laplacian, size_x, wsi_name_test = prepare_data.prepare_test_data(DATA_NAME, DATASET_NAME)
    # labels = prepare_data.retrieve_survival_labels_train(DATA_NAME, DATASET_NAME, LABEL_DIR)
    #
    # tets_labels = prepare_data.retrieve_survival_labels_test(DATA_NAME, DATASET_NAME, LABEL_DIR)
    #
    # test_status, test_time, train_pid = find_labels(wsi_name_train, labels, dataset=DATA_NAME)
    # test_status, test_time, test_pid = find_labels(wsi_name_test, tets_labels, dataset=DATA_NAME)
    # print(np.unique(train_pid).shape)
    # print(np.unique(test_pid).shape)
    # count_pid(train_data)