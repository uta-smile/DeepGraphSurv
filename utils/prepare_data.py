
import numpy as np
import os
import sys
import csv

BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models/'))


import WSI_NLST_loader
import WSI_TCGA_loader
import WSI_GBM_loader

BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf')
NLST_DATA_DIR = os.path.join(BASE_DIR, 'data/WSI/NLST_128')
TCGA_DATA_DIR = os.path.join(BASE_DIR, 'data/WSI/TCGA_128')
TCGA_GBM_DATA_DIR = os.path.join(BASE_DIR, 'data/WSI/TCGA_GBM_128')


def prepare_train_data(data_comb, dataset_name):
    """ data comb is the name for data we used,
        it could be TCGA-ALL, NLST-ALL, TCGA-LUSC, TCGA-LUAD, ACC, SCC, GBM

    """
    tcga_train, _, max_node = WSI_TCGA_loader.load(dataset_name)
    nlst_train, _, max_node = WSI_NLST_loader.load(dataset_name)

    if data_comb == 'TCGA':
        data = tcga_train['X']
        laplacian = tcga_train['L']
        size_x = tcga_train['size']
        wsi_name = tcga_train['name']

    elif data_comb == 'TCGA-LUSC':
        train_data = tcga_train['X']
        train_label = tcga_train['y']
        train_laplacian = tcga_train['L']
        train_size_x = tcga_train['size']
        train_wsi_name = tcga_train['name']

        """ split the data by disease"""
        train_data_lusc = train_data[train_label == 1]
        train_lap_lusc = train_laplacian[train_label == 1]
        train_size_x_lusc = train_size_x[train_label == 1]
        train_wsi_name_lusc = train_wsi_name[train_label == 1]

        data = train_data_lusc
        laplacian = train_lap_lusc
        size_x = train_size_x_lusc
        wsi_name = train_wsi_name_lusc

    elif data_comb == 'TCGA-LUAD':
        train_data = tcga_train['X']
        train_label = tcga_train['y']
        train_laplacian = tcga_train['L']
        train_size_x = tcga_train['size']
        train_wsi_name = tcga_train['name']

        train_data_luad = train_data[train_label == 0]
        train_lap_luad = train_laplacian[train_label == 0]
        train_size_x_luad = train_size_x[train_label == 0]
        train_wsi_name_luad = train_wsi_name[train_label == 0]

        data = train_data_luad
        laplacian = train_lap_luad
        size_x = train_size_x_luad
        wsi_name = train_wsi_name_luad

    elif data_comb == 'NLST':
        data = nlst_train['X']
        laplacian = nlst_train['L']
        size_x = nlst_train['size']
        wsi_name = nlst_train['name']
        node_img = nlst_train['node_img']

    elif data_comb == 'ADC':
        tcga_data = tcga_train['X']
        tcga_label = tcga_train['y']
        tcga_laplacian = tcga_train['L']
        tcga_size_x = tcga_train['size']
        tcga_wsi_name = tcga_train['name']

        tcga_data_luad = tcga_data[tcga_label == 0]
        tcga_lap_luad = tcga_laplacian[tcga_label == 0]
        tcga_size_x_luad = tcga_size_x[tcga_label == 0]
        tcga_wsi_name_luad = tcga_wsi_name[tcga_label == 0]

        nlst_data = nlst_train['X']
        nlst_label = nlst_train['y']
        nlst_laplacian = nlst_train['L']
        nlst_size_x = nlst_train['size']
        nlst_wsi_name = nlst_train['name']

        nlst_data_luad = nlst_data[nlst_label == 0]
        nlst_lap_luad = nlst_laplacian[nlst_label == 0]
        nlst_size_x_luad = nlst_size_x[nlst_label == 0]
        nlst_wsi_name_luad = nlst_wsi_name[nlst_label == 0]

        data = np.concatenate((tcga_data_luad, nlst_data_luad), axis=0)
        laplacian = np.concatenate((tcga_lap_luad, nlst_lap_luad), axis=0)
        size_x = np.concatenate((tcga_size_x_luad, nlst_size_x_luad), axis=0)
        wsi_name = np.concatenate((tcga_wsi_name_luad, nlst_wsi_name_luad), axis=0)

    elif data_comb == 'SCC':

        tcga_data = tcga_train['X']
        tcga_label = tcga_train['y']
        tcga_laplacian = tcga_train['L']
        tcga_size_x = tcga_train['size']
        tcga_wsi_name = tcga_train['name']

        tcga_data_lusc = tcga_data[tcga_label == 1]
        tcga_lap_lusc = tcga_laplacian[tcga_label == 1]
        tcga_size_x_lusc = tcga_size_x[tcga_label == 1]
        tcga_wsi_name_lusc = tcga_wsi_name[tcga_label == 1]

        nlst_data = nlst_train['X']
        nlst_label = nlst_train['y']
        nlst_laplacian = nlst_train['L']
        nlst_size_x = nlst_train['size']
        nlst_wsi_name = nlst_train['name']

        nlst_data_lusc = nlst_data[nlst_label == 1]
        nlst_lap_lusc = nlst_laplacian[nlst_label == 1]
        nlst_size_x_lusc = nlst_size_x[nlst_label == 1]
        nlst_wsi_name_lusc = nlst_wsi_name[nlst_label == 1]

        data = np.concatenate((tcga_data_lusc, nlst_data_lusc), axis=0)
        laplacian = np.concatenate((tcga_lap_lusc, nlst_lap_lusc), axis=0)
        size_x = np.concatenate((tcga_size_x_lusc, nlst_size_x_lusc), axis=0)
        wsi_name = np.concatenate((tcga_wsi_name_lusc, nlst_wsi_name_lusc), axis=0)
    elif data_comb == 'TCGA-GBM':

        gbm_train, _, max_node = WSI_GBM_loader.load(dataset_name)
        data = gbm_train['X']
        laplacian = gbm_train['L']
        size_x = gbm_train['size']
        wsi_name = gbm_train['name']
    else:
        ValueError('No Such Data Comb %s' % data_comb)

    return data, laplacian, size_x, wsi_name, node_img


def prepare_test_data(data_comb, dataset_name):
    """ data comb is the name for data we used,
        it could be TCGA-ALL, NLST-ALL, TCGA-LUSC, TCGA-LUAD, ACC, SCC, GBM
        dataset_name : e.g processed_500maxnode
        for TCGA and NLSt, we use same max-node set for both
    """
    _, tcga_test, max_node = WSI_TCGA_loader.load(dataset_name)
    _, nlst_test, max_node = WSI_NLST_loader.load(dataset_name)

    if data_comb == 'TCGA':
        data = tcga_test['X']
        laplacian = tcga_test['L']
        size_x = tcga_test['size']
        wsi_name = tcga_test['name']

    elif data_comb == 'TCGA-LUSC':
        train_data = tcga_test['X']
        train_label = tcga_test['y']
        train_laplacian = tcga_test['L']
        train_size_x = tcga_test['size']
        train_wsi_name = tcga_test['name']

        """ split the data by disease"""
        train_data_lusc = train_data[train_label == 1]
        train_lap_lusc = train_laplacian[train_label == 1]
        train_size_x_lusc = train_size_x[train_label == 1]
        train_wsi_name_lusc = train_wsi_name[train_label == 1]

        data = train_data_lusc
        laplacian = train_lap_lusc
        size_x = train_size_x_lusc
        wsi_name = train_wsi_name_lusc

    elif data_comb == 'TCGA-LUAD':
        train_data = tcga_test['X']
        train_label = tcga_test['y']
        train_laplacian = tcga_test['L']
        train_size_x = tcga_test['size']
        train_wsi_name = tcga_test['name']

        train_data_luad = train_data[train_label == 0]
        train_lap_luad = train_laplacian[train_label == 0]
        train_size_x_luad = train_size_x[train_label == 0]
        train_wsi_name_luad = train_wsi_name[train_label == 0]

        data = train_data_luad
        laplacian = train_lap_luad
        size_x = train_size_x_luad
        wsi_name = train_wsi_name_luad

    elif data_comb == 'NLST':
        data = nlst_test['X']
        laplacian = nlst_test['L']
        size_x = nlst_test['size']
        wsi_name = nlst_test['name']

    elif data_comb == 'ADC':
        tcga_data = tcga_test['X']
        tcga_label = tcga_test['y']
        tcga_laplacian = tcga_test['L']
        tcga_size_x = tcga_test['size']
        tcga_wsi_name = tcga_test['name']

        tcga_data_luad = tcga_data[tcga_label == 0]
        tcga_lap_luad = tcga_laplacian[tcga_label == 0]
        tcga_size_x_luad = tcga_size_x[tcga_label == 0]
        tcga_wsi_name_luad = tcga_wsi_name[tcga_label == 0]

        nlst_data = nlst_test['X']
        nlst_label = nlst_test['y']
        nlst_laplacian = nlst_test['L']
        nlst_size_x = nlst_test['size']
        nlst_wsi_name = nlst_test['name']

        nlst_data_luad = nlst_data[nlst_label == 0]
        nlst_lap_luad = nlst_laplacian[nlst_label == 0]
        nlst_size_x_luad = nlst_size_x[nlst_label == 0]
        nlst_wsi_name_luad = nlst_wsi_name[nlst_label == 0]

        data = np.concatenate((tcga_data_luad, nlst_data_luad), axis=0)
        laplacian = np.concatenate((tcga_lap_luad, nlst_lap_luad), axis=0)
        size_x = np.concatenate((tcga_size_x_luad, nlst_size_x_luad), axis=0)
        wsi_name = np.concatenate((tcga_wsi_name_luad, nlst_wsi_name_luad), axis=0)

    elif data_comb == 'SCC':

        tcga_data = tcga_test['X']
        tcga_label = tcga_test['y']
        tcga_laplacian = tcga_test['L']
        tcga_size_x = tcga_test['size']
        tcga_wsi_name = tcga_test['name']

        tcga_data_lusc = tcga_data[tcga_label == 1]
        tcga_lap_lusc = tcga_laplacian[tcga_label == 1]
        tcga_size_x_lusc = tcga_size_x[tcga_label == 1]
        tcga_wsi_name_lusc = tcga_wsi_name[tcga_label == 1]

        nlst_data = nlst_test['X']
        nlst_label = nlst_test['y']
        nlst_laplacian = nlst_test['L']
        nlst_size_x = nlst_test['size']
        nlst_wsi_name = nlst_test['name']

        nlst_data_lusc = nlst_data[nlst_label == 1]
        nlst_lap_lusc = nlst_laplacian[nlst_label == 1]
        nlst_size_x_lusc = nlst_size_x[nlst_label == 1]
        nlst_wsi_name_lusc = nlst_wsi_name[nlst_label == 1]

        data = np.concatenate((tcga_data_lusc, nlst_data_lusc), axis=0)
        laplacian = np.concatenate((tcga_lap_lusc, nlst_lap_lusc), axis=0)
        size_x = np.concatenate((tcga_size_x_lusc, nlst_size_x_lusc), axis=0)
        wsi_name = np.concatenate((tcga_wsi_name_lusc, nlst_wsi_name_lusc), axis=0)

    elif data_comb == 'TCGA-GBM':
        _, gbm_test, max_node = WSI_GBM_loader.load(dataset_name)
        data = gbm_test['X']
        laplacian = gbm_test['L']
        size_x = gbm_test['size']
        wsi_name = gbm_test['name']
    else:
        ValueError('No Such Data Comb %s' % data_comb)

    return data, laplacian, size_x, wsi_name


def parse_csv(csv_dir, dataset):
    import csv
    pid = []
    status = []
    time = []
    if dataset == 'TCGA':
        with open(csv_dir) as csvDataFile:
            reader = csv.reader(csvDataFile)
            for row in reader:
                # print(row)
                pid.append(row[0])
                status.append(row[1])
                time.append(row[2])
        return pid[1:], status[1:], time[1:]
    elif dataset == 'NLST':
        wsi_name = []
        with open(csv_dir) as csvDataFile:
            reader = csv.reader(csvDataFile)
            for row in reader:
                # print(row)
                wsi_name.append(row[0])
                pid.append(row[1])
                status.append(row[2])
                time.append(row[3])
        return wsi_name[1:], pid[1:], status[1:], time[1:]


def save_labels_to_csv(labels, data_dir, file_name):
    if os.path.exists(os.path.join(NLST_DATA_DIR, file_name)):
        return

    pid = labels['pid']
    wsi = labels['wsi']
    status = labels['status']
    time = labels['time']
    save_dir = os.path.join(data_dir, file_name)
    with open(save_dir, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['pid', 'wsi_id', 'status', 'survival_time'])
        for p, w, s, t in zip(pid, wsi, status, time):
            writer.writerow([p] + [w] + [s] + [t])

    print("Successfully saved labels at % s" % save_dir)


def retrieve_survival_labels_train(dataset, dataset_name, label_dir, save=False):
    """ get the status/ survival time label """
    tcga_label_train_LUAD = os.path.join(label_dir, 'TCGA_LUAD_Train_1.csv')
    tcga_label_train_LUSC = os.path.join(label_dir, 'TCGA_LUSC_Train_1.csv')
    gbm_label_train = os.path.join(label_dir, 'TCGA_GBM_Train_1.csv')

    nlst_label_train_LUAD = os.path.join(label_dir, 'NLST_ADC_Train_1.csv')
    nlst_label_train_LUSC = os.path.join(label_dir, 'NLST_SCC_Train_1.csv')

    if dataset == 'TCGA' or dataset == 'TCGA-LUAD' or dataset == 'TCGA-LUSC':
        LUAD_pid, LUAD_status, LUAD_time = parse_csv(tcga_label_train_LUAD, 'TCGA')
        LUSC_pid, LUSC_status, LUSC_time = parse_csv(tcga_label_train_LUSC, 'TCGA')

        labels = {
            'pid': LUAD_pid + LUSC_pid,
            'status': LUAD_status + LUSC_status,
            'time': LUAD_time + LUSC_time,
        }

        lusc_label = {
            'pid': LUSC_pid,
            'status': LUSC_status,
            'time': LUSC_time,
        }

        luad_label = {
            'pid': LUAD_pid,
            'status': LUAD_status,
            'time': LUAD_time,
        }

        " adding this part for only making CSV for TCGA TCGA-LUSC, TCGA-LUAD"
        tcga_train, _, max_node = WSI_TCGA_loader.load(dataset_name)

        train_label = tcga_train['y']
        wsi_name = tcga_train['name']

        wsi_status_list, wsi_time_list, wsi_pid_list = [], [], []
        for wsi in wsi_name.tolist():
            wsi_id = wsi[:12]   # pid
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

        labels_tcga_all = {
            'pid': wsi_pid_list,
            'wsi': wsi_name,
            'status': wsi_status_list,
            'time': wsi_time_list,
        }
        if save:
            save_labels_to_csv(labels_tcga_all, TCGA_DATA_DIR, 'TCGA_all_labels_train.csv')

        train_wsi_name_lusc = wsi_name[train_label == 1]
        lusc_wsi_status_list, lusc_wsi_time_list, lusc_wsi_pid_list = [], [], []
        for wsi in train_wsi_name_lusc.tolist():
            wsi_id = wsi[:12]  # pid
            # print("finding labels for WSI %s" % wsi_id)

            if wsi_id in lusc_label['pid']:
                # here wsi_id is pid
                idx = lusc_label['pid'].index(wsi_id)
                status = float(lusc_label['status'][idx])
                time = float(lusc_label['time'][idx])
                pid = wsi_id
                lusc_wsi_status_list.append(status)
                lusc_wsi_time_list.append(time)
                lusc_wsi_pid_list.append(pid)

        labels_tcga_lusc = {
            'pid': lusc_wsi_pid_list,
            'wsi': train_wsi_name_lusc,
            'status': lusc_wsi_status_list,
            'time': lusc_wsi_time_list,
        }
        if save:
            save_labels_to_csv(labels_tcga_lusc, TCGA_DATA_DIR, 'TCGA-LUSC_labels_train.csv')

        train_wsi_name_luad = wsi_name[train_label == 0]
        luad_wsi_status_list, luad_wsi_time_list, luad_wsi_pid_list = [], [], []
        for wsi in train_wsi_name_luad.tolist():
            wsi_id = wsi[:12]  # pid
            # print("finding labels for WSI %s" % wsi_id)

            if wsi_id in luad_label['pid']:
                # here wsi_id is pid
                idx = luad_label['pid'].index(wsi_id)
                status = float(luad_label['status'][idx])
                time = float(luad_label['time'][idx])
                pid = wsi_id
                luad_wsi_status_list.append(status)
                luad_wsi_time_list.append(time)
                luad_wsi_pid_list.append(pid)

        labels_tcga_luad = {
            'pid': luad_wsi_pid_list,
            'wsi': train_wsi_name_luad,
            'status': luad_wsi_status_list,
            'time': luad_wsi_time_list,
        }
        if save:
            save_labels_to_csv(labels_tcga_luad, TCGA_DATA_DIR, 'TCGA-LUAD_labels_train.csv')

        return labels

    elif dataset == 'NLST':
        LUAD_wsi, LUAD_pid, LUAD_status, LUAD_time = parse_csv(nlst_label_train_LUAD, 'NLST')
        LUSC_wsi, LUSC_pid, LUSC_status, LUSC_time = parse_csv(nlst_label_train_LUSC, 'NLST')

        labels = {
            'pid': LUAD_pid + LUSC_pid,
            'status': LUAD_status + LUSC_status,
            'time': LUAD_time + LUSC_time,
            'wsi': LUAD_wsi + LUSC_wsi
        }
        if save:
            save_labels_to_csv(labels, NLST_DATA_DIR, 'NLST_ADCSCC_labels_train.csv')

        return labels

    elif dataset == 'ADC':
        tcga_LUAD_pid, tcga_LUAD_status, tcga_LUAD_time = parse_csv(tcga_label_train_LUAD, 'TCGA')
        nlst_LUAD_wsi, nlst_LUAD_pid, nlst_LUAD_status, nlst_LUAD_time = parse_csv(nlst_label_train_LUAD, 'NLST')

        tcga_labels = {
            'pid': tcga_LUAD_pid,
            'status': tcga_LUAD_status,
            'time': tcga_LUAD_time,
        }
        nlst_labels = {
            'pid': nlst_LUAD_pid,
            'status': nlst_LUAD_status,
            'time': nlst_LUAD_time,
            'wsi': nlst_LUAD_wsi
        }
        "because TCGA NLST data structure differs, we return two labels dict"
        return nlst_labels, tcga_labels

    elif dataset == 'SCC':
        tcga_LUAD_pid, tcga_LUAD_status, tcga_LUAD_time = parse_csv(tcga_label_train_LUSC, 'TCGA')
        nlst_LUAD_wsi, nlst_LUAD_pid, nlst_LUAD_status, nlst_LUAD_time = parse_csv(nlst_label_train_LUSC, 'NLST')

        tcga_labels = {
            'pid': tcga_LUAD_pid,
            'status': tcga_LUAD_status,
            'time': tcga_LUAD_time,
        }
        nlst_labels = {
            'pid': nlst_LUAD_pid,
            'status': nlst_LUAD_status,
            'time': nlst_LUAD_time,
            'wsi': nlst_LUAD_wsi
        }
        "because TCGA NLST data structure differs, we return two labels dict"
        return nlst_labels, tcga_labels

    elif dataset == 'TCGA-GBM':
        gbm_pid, gbm_status, gbm_time = parse_csv(gbm_label_train, 'TCGA')
        labels = {
            'pid':gbm_pid,
            'status': gbm_status,
            'time': gbm_time,
        }
        if save:
            save_labels_to_csv(labels, TCGA_GBM_DATA_DIR, 'GBM_labels_train.csv')

        return labels


def retrieve_survival_labels_test(dataset, dataset_name, label_dir, save=False):
    """ get the status/ survival time label """

    tcga_label_test_LUAD = os.path.join(label_dir, 'TCGA_LUAD_Test_1.csv')
    tcga_label_test_LUSC = os.path.join(label_dir, 'TCGA_LUSC_Test_1.csv')
    gbm_label_test = os.path.join(label_dir, 'TCGA_GBM_Test_1.csv')

    nlst_label_test_LUAD = os.path.join(label_dir, 'NLST_ADC_Test_1.csv')
    nlst_label_test_LUSC = os.path.join(label_dir, 'NLST_SCC_Test_1.csv')

    if dataset == 'TCGA' or dataset == 'TCGA-LUAD' or dataset == 'TCGA-LUSC':
        LUAD_pid, LUAD_status, LUAD_time = parse_csv(tcga_label_test_LUAD, 'TCGA')
        LUSC_pid, LUSC_status, LUSC_time = parse_csv(tcga_label_test_LUSC, 'TCGA')

        labels = {
            'pid': LUAD_pid + LUSC_pid,
            'status': LUAD_status + LUSC_status,
            'time': LUAD_time + LUSC_time,
        }

        lusc_label = {
            'pid': LUSC_pid,
            'status': LUSC_status,
            'time': LUSC_time,
        }

        luad_label = {
            'pid': LUAD_pid,
            'status': LUAD_status,
            'time': LUAD_time,
        }

        " adding this part for only making CSV for TCGA TCGA-LUSC, TCGA-LUAD"
        _, tcga_test, max_node = WSI_TCGA_loader.load(dataset_name)

        train_label = tcga_test['y']
        wsi_name = tcga_test['name']

        wsi_status_list, wsi_time_list, wsi_pid_list = [], [], []
        for wsi in wsi_name.tolist():
            wsi_id = wsi[:12]  # pid
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

        labels_tcga_all = {
            'pid': wsi_pid_list,
            'wsi': wsi_name,
            'status': wsi_status_list,
            'time': wsi_time_list,
        }
        if save:
            save_labels_to_csv(labels_tcga_all, TCGA_DATA_DIR, 'TCGA_all_labels_test.csv')

        train_wsi_name_lusc = wsi_name[train_label == 1]
        lusc_wsi_status_list, lusc_wsi_time_list, lusc_wsi_pid_list = [], [], []
        for wsi in train_wsi_name_lusc.tolist():
            wsi_id = wsi[:12]  # pid
            # print("finding labels for WSI %s" % wsi_id)

            if wsi_id in lusc_label['pid']:
                # here wsi_id is pid
                idx = lusc_label['pid'].index(wsi_id)
                status = float(lusc_label['status'][idx])
                time = float(lusc_label['time'][idx])
                pid = wsi_id
                lusc_wsi_status_list.append(status)
                lusc_wsi_time_list.append(time)
                lusc_wsi_pid_list.append(pid)

        labels_tcga_lusc = {
            'pid': lusc_wsi_pid_list,
            'wsi': train_wsi_name_lusc,
            'status': lusc_wsi_status_list,
            'time': lusc_wsi_time_list,
        }
        if save:
            save_labels_to_csv(labels_tcga_lusc, TCGA_DATA_DIR, 'TCGA-LUSC_labels_test.csv')

        train_wsi_name_luad = wsi_name[train_label == 0]
        luad_wsi_status_list, luad_wsi_time_list, luad_wsi_pid_list = [], [], []
        for wsi in train_wsi_name_luad.tolist():
            wsi_id = wsi[:12]  # pid
            # print("finding labels for WSI %s" % wsi_id)

            if wsi_id in luad_label['pid']:
                # here wsi_id is pid
                idx = luad_label['pid'].index(wsi_id)
                status = float(luad_label['status'][idx])
                time = float(luad_label['time'][idx])
                pid = wsi_id
                luad_wsi_status_list.append(status)
                luad_wsi_time_list.append(time)
                luad_wsi_pid_list.append(pid)

        labels_tcga_luad = {
            'pid': luad_wsi_pid_list,
            'wsi': train_wsi_name_luad,
            'status': luad_wsi_status_list,
            'time': luad_wsi_time_list,
        }
        if save:
            save_labels_to_csv(labels_tcga_luad, TCGA_DATA_DIR, 'TCGA-LUAD_labels_test.csv')

        return labels

    elif dataset == 'NLST':
        LUAD_wsi, LUAD_pid, LUAD_status, LUAD_time = parse_csv(nlst_label_test_LUAD, 'NLST')
        LUSC_wsi, LUSC_pid, LUSC_status, LUSC_time = parse_csv(nlst_label_test_LUSC, 'NLST')

        labels = {
            'pid': LUAD_pid + LUSC_pid,
            'status': LUAD_status + LUSC_status,
            'time': LUAD_time + LUSC_time,
            'wsi': LUAD_wsi + LUSC_wsi
        }
        if save:
            save_labels_to_csv(labels, NLST_DATA_DIR, 'NLST_ADCSCC_labels_test.csv')

        return labels

    elif dataset == 'ADC':
        tcga_LUAD_pid, tcga_LUAD_status, tcga_LUAD_time = parse_csv(tcga_label_test_LUAD, 'TCGA')
        nlst_LUAD_wsi, nlst_LUAD_pid, nlst_LUAD_status, nlst_LUAD_time = parse_csv(nlst_label_test_LUAD, 'NLST')

        tcga_labels = {
            'pid': tcga_LUAD_pid,
            'status': tcga_LUAD_status,
            'time': tcga_LUAD_time,
        }
        nlst_labels = {
            'pid': nlst_LUAD_pid,
            'status': nlst_LUAD_status,
            'time': nlst_LUAD_time,
            'wsi': nlst_LUAD_wsi
        }
        "because TCGA NLST data structure differs, we return two labels dict"
        return nlst_labels, tcga_labels
    elif dataset == 'SCC':
        tcga_LUAD_pid, tcga_LUAD_status, tcga_LUAD_time = parse_csv(tcga_label_test_LUSC, 'TCGA')
        nlst_LUAD_wsi, nlst_LUAD_pid, nlst_LUAD_status, nlst_LUAD_time = parse_csv(nlst_label_test_LUSC, 'NLST')

        tcga_labels = {
            'pid': tcga_LUAD_pid,
            'status': tcga_LUAD_status,
            'time': tcga_LUAD_time,
        }
        nlst_labels = {
            'pid': nlst_LUAD_pid,
            'status': nlst_LUAD_status,
            'time': nlst_LUAD_time,
            'wsi': nlst_LUAD_wsi
        }
        "because TCGA NLST data structure differs, we return two labels dict"
        return nlst_labels, tcga_labels
    elif dataset == 'TCGA-GBM':
        gbm_pid, gbm_status, gbm_time = parse_csv(gbm_label_test, 'TCGA')
        labels = {
            'pid':gbm_pid,
            'status': gbm_status,
            'time': gbm_time,
        }
        if save:
            save_labels_to_csv(labels, TCGA_GBM_DATA_DIR, 'GBM_labels_test.csv')

        return labels


if __name__ == "__main__":
    dataset_name = 'processed_500maxnode'
    label_dir = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf', 'models/survival_labels')
    retrieve_survival_labels_train('TCGA-LUSC', dataset_name, label_dir)
