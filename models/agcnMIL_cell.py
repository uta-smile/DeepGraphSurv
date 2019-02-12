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
import glob

import networkx as nx
from sklearn.metrics import roc_curve, roc_auc_score
from lifelines.utils import concordance_index


BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

DATA_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/data/cell/Patches')

import io_utils


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='AGCN_MIL',
                    help='Model name: AGCN_MIL[default: AGCN_MIL]')
parser.add_argument('--log_dir', default='log/agcnMIL_cls_cell', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=500, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=2e3, help='Decay step for lr decay [default: 200000]')
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
NUM_FEATURE = 3
HEIGHT = 27
WIDTH = 27
NUM_CLASSES = 2
EVAl_FREQ = 5

MODEL_NAME = 'agcnMIL'
DATA_NAME = 'cell'

parameter_pack = {
    'lr': BASE_LEARNING_RATE,
    'max_node': MAX_NUM_POINT,
    'batch_size': BATCH_SIZE,
}

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')


BASE_LOG_DIR = FLAGS.log_dir
if not os.path.exists(BASE_LOG_DIR):
    os.mkdir(BASE_LOG_DIR)

LOG_DIR = os.path.join(BASE_LOG_DIR, 'node_%s_batch_%s' % (MAX_NUM_POINT, BATCH_SIZE))
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

RESULT_DIR = os.path.join(LOG_DIR, 'results')
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

IMG_DIR = os.path.join(LOG_DIR, 'image_save')
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

MODEL_SAVE_DIR = os.path.join(LOG_DIR, 'model_save')
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)


LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_%s_cls_%s.txt' % (MODEL_NAME, DATASET_NAME)), 'w')
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
    model_list = ['agcn']
    for model_name in model_list:
        print("working on model %s" % model_name)

        tf.set_random_seed(123)

        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(GPU_INDEX)):
                image_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, MAX_NUM_POINT, HEIGHT, WIDTH)
                is_training_pl = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)

                # Get model and loss
                if model_name == 'agcn':
                    end_points = MODEL.agcnMIL_wsi_cls(image_pl, NUM_CLASSES, is_training_pl, bn_decay)
                    loss = MODEL.cross_entropy(end_points, labels_pl)

                else:
                    ValueError("No Such Model Name{}".format(model_name))

                # loss_reg = MODEL.loss_reg(end_points['regs_list'])

                # Get training operator
                # base_lr = 1e-4
                # learning_rate = get_learning_rate(batch, base_lr)
                # tf.summary.scalar('learning_rate', learning_rate)
                learning_rate_pl = tf.placeholder(tf.float32, shape=())

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl,
                                                   beta1=0.9, beta2=0.999,)
                train_op = optimizer.minimize(loss, global_step=batch)

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Init variables
            # init = tf.global_variables_initializer()

            ops = {'image_pl': image_pl,
                   'label_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'learning_rate_pl': learning_rate_pl,
                   'pred': end_points['class_prediction'],
                   'loss': loss,
                   'weight_loss': end_points['weight_decay'],
                   'attn': end_points['attn'],
                   'watch_1': end_points['ss'],
                   'train_op': train_op,
                   'saver_op': end_points['saver'],
                   'step': batch}

            # train model
            max_repeat = 1

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init,
                     {is_training_pl: True})

            # cross_validation = []
            loss_seq = [0.0] * MAX_EPOCH
            for lr in [1e-4]:
            # for _ in range(max_repeat):

                loader = cell_img_loader()
                for epoch in range(MAX_EPOCH):
                    # log_string('**** EPOCH %03d ****' % epoch)
                    sys.stdout.flush()

                    loss = train_one_epoch(epoch, sess, ops, lr, loader)
                    loss_seq[epoch] += loss

                    if epoch > 0 and epoch % 5 == 0:
                        evaluate_one_epoch(epoch, sess, ops, loader)
                        # save_feature_csv(train_data_dict, test_data_dict, epoch)

                """ train on LASSO COX using the latest predictions"""
                # train_cox_lasso(train_data_dict, test_data_dict)
                # save_feature_csv(train_data_dict, test_data_dict)
                # best_set = statistic_results(c_index_list, p_value_list)
                # cross_validation.append(best_set)

            # calculate , print CI after cross-validation
            # process_cv(cross_validation, parameter_pack, trial_id)
            # cal and save the  (avg.) loss sequence for model comparison
            save_loss_seq([l / float(MAX_EPOCH) for l in loss_seq], parameter_pack)


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


def train_one_epoch(epoch, sess, ops, lr, loader):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    # train, _, max_node = loader.load_train()
    #
    # train_data = train['X']
    # train_label = train['y']
    #
    # file_size = train_data.shape[0]
    # # num_batches = file_size // BATCH_SIZE
    # batch_count = 0
    # start_idx = 0
    # end_idx = BATCH_SIZE

    # lr_list = [1e-4, 5e-4, 1e-4, 8e-4, 5e-5, 5e-6]
    # lr = lr_list[0]

    loss_sum = 0.0
    correct_total = 0
    seen_total = 0

    while not loader.end_train:
        batch_data, batch_label = loader.load_train()

        feed_dict = {ops['image_pl']: batch_data,
                     ops['is_training_pl']: is_training,
                     ops['learning_rate_pl']: lr,
                     ops['label_pl']: batch_label,
                     }
        step, _, loss, weight_loss, pred, attn, fea, lr = sess.run(
            [
                ops['step'],
                ops['train_op'],
                ops['loss'],
                ops['weight_loss'],
                ops['pred'],
                ops['attn'],
                ops['watch_1'],
                ops['learning_rate_pl'],
            ],
            feed_dict=feed_dict)
        sum_attn = np.squeeze(np.sum(attn))
        loss_sum += loss
        predicted_class = np.round(pred)
        correct = np.sum(predicted_class == batch_label)
        correct_total += correct
        seen_total += BATCH_SIZE
        # start_idx = end_idx
        # end_idx += BATCH_SIZE
    print('learning rate %f' % lr)

    avg_loss = loss_sum
    log_string('accuracy at epoch %s: %f' % (epoch, correct_total / float(seen_total)))
    log_string('mean loss at epoch %s: %f \n' % (epoch, avg_loss))

    if epoch > 19 and epoch % 10 == 0:
        saver = ops['saver_op']
        save_path = saver.save(sess, MODEL_SAVE_DIR + '/agcn_survival_backbone_ep{}'.format(epoch))
        print("Backbone Net saved in file: %s" % save_path)

        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

        print_tensors_in_checkpoint_file(file_name=save_path,
                                         tensor_name='',
                                         all_tensors=False)
    loader.restart_train_epoch()

    return avg_loss


def evaluate_one_epoch(epoch, sess, ops, loader):
    is_training = False
    # _, test, max_node = loader.load_test()
    #
    # test_data = test['X']
    # test_label = test['y']
    # file_size = test_data.shape[0]
    # num_batches = file_size // BATCH_SIZE

    correct_total = 0
    seen_total = 0

    # for batch_idx in range(num_batches):
    while not loader.end_test:

        batch_data, batch_label = loader.load_test()

        # s = np.arange(batch_data.shape[0])
        # np.random.shuffle(s)
        # batch_data = batch_data[s]
        # batch_label = batch_label[s]

        feed_dict = {ops['image_pl']: batch_data,
                     ops['is_training_pl']: is_training,
                     }
        pred = sess.run(ops['pred'],
                        feed_dict=feed_dict)

        predicted_class = np.round(pred)
        correct = np.sum(predicted_class == batch_label)
        correct_total += correct
        seen_total += BATCH_SIZE
    #
    # left_num = file_size - end_idx
    # pad_num = BATCH_SIZE - left_num
    # if pad_num > 0 and left_num > 0:
    #     last_batch_data = np.concatenate((test_data[-left_num:], test_data[: pad_num]), axis=0)
    #
    #     feed_dict = {ops['image_pl']: last_batch_data,
    #                  ops['is_training_pl']: is_training,
    #                  }
    #     pred, graph_feature = sess.run(
    #         [
    #             ops['pred'],
    #             ops['graph_feature']
    #         ],
    #         feed_dict=feed_dict)
    #     predicted_class = np.argmax(pred[: left_num], axis=1)
    #     correct = np.sum(predicted_class == test_data[-left_num:])
    #     correct_total += correct
    #     seen_total += BATCH_SIZE
    loader.restart_test_epoch()
    log_string('TESTING: accuracy at epoch %s: %f \n\n' % (epoch, correct_total / float(seen_total)))


class cell_img_loader(object):

    def __init__(self, data_dir=DATA_DIR, cv=True):
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self.do_cv = cv

        class_0_dir = os.path.join(data_dir, 'class_0')
        class_0_patch_list = [os.path.join(class_0_dir, o) for o in os.listdir(class_0_dir)
                              if os.path.isdir(os.path.join(class_0_dir, o))]

        class_1_dir = os.path.join(data_dir, 'class_1')
        class_1_patch_list = [os.path.join(class_1_dir, o) for o in os.listdir(class_1_dir)
                              if os.path.isdir(os.path.join(class_1_dir, o))]

        all_patch_name_list = class_0_patch_list + class_1_patch_list
        label_list = [0]*len(class_0_patch_list) + [1]*len(class_1_patch_list)

        if self.do_cv:
            s = np.arange(len(all_patch_name_list))
            np.random.shuffle(s)
            patch_list_shuffled = np.array(all_patch_name_list)[s]
            label_list_shuffled = np.array(label_list)[s]
        else:
            patch_list_shuffled = np.array(all_patch_name_list)
            label_list_shuffled = np.array(label_list)

        train_ratio = 0.7
        self.train_patch = patch_list_shuffled[:int(train_ratio * patch_list_shuffled.shape[0])]
        self.train_label = label_list_shuffled[:int(train_ratio * patch_list_shuffled.shape[0])]
        self.test_patch = patch_list_shuffled[int(train_ratio * patch_list_shuffled.shape[0]):]
        self.test_label = label_list_shuffled[int(train_ratio * patch_list_shuffled.shape[0]):]

        n_class_1 = np.sum(self.test_label == 1)
        print("ratio of class 1 %f" % (n_class_1 / self.test_label.shape[0]))
        n_class_0 = np.sum(self.test_label == 0)
        print("ratio of class 0 %f" % (n_class_0 / self.test_label.shape[0]))

        # prepare train:
        patch, cell_name_patch = [], []
        for p in self.train_patch:
            all_cell = []
            cell_labels = []
            for img in glob.glob(p + '/*.bmp'):
                cell_label = img[:-4].split('-')[-1]
                cell_labels.append(cell_label)
                img_tensor = read_img(img)
                all_cell.append(img_tensor)
            all_cell = np.stack(all_cell)
            cell_labels = np.array(cell_labels)
            patch += [all_cell]
            cell_name_patch += [cell_labels]
        self.cells_on_patch = patch
        self.cells_name_on_patch = cell_name_patch

        patch_test, cell_name_patch_test = [], []
        for p in self.test_patch:
            all_cell = []
            cell_labels = []
            for img in glob.glob(p + '/*.bmp'):
                cell_label = img[:-4].split('-')[-1]
                cell_labels.append(cell_label)
                img_tensor = read_img(img)
                all_cell.append(img_tensor)
            all_cell = np.stack(all_cell)
            cell_labels = np.array(cell_labels)
            patch_test += [all_cell]
            cell_name_patch_test += [cell_labels]
        self.cells_on_patch_test = patch_test
        self.cells_name_on_patch_test = cell_name_patch_test

        self.current_idx_train = 0
        self.current_idx_test = 0

        self.num_train = len(self.cells_on_patch)
        self.num_test = len(self.cells_on_patch_test)

        self.end_train = False
        self.end_test = False

    def load_train(self):
        # each call we return one sample
        img = self.cells_on_patch[self.current_idx_train]
        label = self.train_label[self.current_idx_train]
        self.current_idx_train += 1
        if self.current_idx_train >= self.num_train:
            self.end_train = True
        return np.divide(img, 255.0), np.array([label]).astype(np.float32)

    def load_test(self):
        img = self.cells_on_patch_test[self.current_idx_test]
        label = self.test_label[self.current_idx_test]
        self.current_idx_test += 1
        if self.current_idx_test >= self.num_test:
            self.end_test = True
        return np.divide(img, 255.0), np.array([label]).astype(np.float32)

    def restart_train_epoch(self):
        self.end_train = False
        self.current_idx_train = 0

    def restart_test_epoch(self):
        self.end_test = False
        self.current_idx_test = 0


def read_img(img_dir):
    from PIL import Image
    im = Image.open(img_dir)
    p = np.array(im).astype(np.float32)
    return p


if __name__ == "__main__":
    train()
    # loader = cell_img_loader()
    # d, l = loader.load_train()
    # print(d)