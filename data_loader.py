import scipy.io as sio
import math
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing


def load_dataset(Dataset, is_pca):
    data_path = './Datasets/'
    if Dataset == 'IN':
        data_path = data_path + 'IndianPines/'
        mat_data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')
        mat_gt = sio.loadmat(data_path + 'Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        # 总共21025个像素。10249是地物像素，其余10776是背景像素（需要剔除），除去背景共16类
        K = 30

    elif Dataset == 'PU':
        data_path = data_path + 'PaviaU/'
        uPavia = sio.loadmat(data_path + 'PaviaU.mat')
        gt_uPavia = sio.loadmat(data_path + 'PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        K = 15

    elif Dataset == 'PC':
        data_path = data_path + 'PaviaC/'
        uPavia = sio.loadmat(data_path + 'Pavia.mat')
        gt_uPavia = sio.loadmat(data_path + 'Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        K = 15

    elif Dataset == 'B':
        data_path = data_path + 'Botswana/'
        uPavia = sio.loadmat(data_path + 'Botswana.mat')
        gt_uPavia = sio.loadmat(data_path + 'Botswana_gt.mat')
        data_hsi = uPavia['Botswana']
        gt_hsi = gt_uPavia['Botswana_gt']
        K = 15

    elif Dataset == 'SC':
        data_path = data_path + 'Salinas/'
        SV = sio.loadmat(data_path + 'Salinas_corrected.mat')
        gt_SV = sio.loadmat(data_path + 'Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        K = 15

    elif Dataset == 'KSC':
        data_path = data_path + 'KSC/'
        SV = sio.loadmat(data_path + 'KSC.mat')
        gt_SV = sio.loadmat(data_path + 'KSC_gt.mat')
        data_hsi = SV['KSC']
        gt_hsi = gt_SV['KSC_gt']
        K = 15

    elif Dataset == 'HC':
        data_path = data_path + 'WHU-Hi-HanChuan/'
        SV = sio.loadmat(data_path + 'WHU_Hi_HanChuan.mat')
        gt_SV = sio.loadmat(data_path + 'WHU_Hi_HanChuan_gt.mat')
        data_hsi = SV['WHU_Hi_HanChuan']
        gt_hsi = gt_SV['WHU_Hi_HanChuan_gt']
        K = 15

    elif Dataset == 'HH':
        data_path = data_path + 'WHU-Hi-HongHu/'
        SV = sio.loadmat(data_path + 'WHU_Hi_HongHu.mat')
        gt_SV = sio.loadmat(data_path + 'WHU_Hi_HongHu_gt.mat')
        data_hsi = SV['WHU_Hi_HongHu']
        gt_hsi = gt_SV['WHU_Hi_HongHu_gt']
        K = 30

    elif Dataset == 'LK':
        data_path = data_path + 'WHU-Hi-LongKou/'
        SV = sio.loadmat(data_path + 'WHU_Hi_LongKou.mat')
        gt_SV = sio.loadmat(data_path + 'WHU_Hi_LongKou_gt.mat')
        data_hsi = SV['WHU_Hi_LongKou']
        gt_hsi = gt_SV['WHU_Hi_LongKou_gt']
        K = 15

    else:
        raise NotImplementedError('Dataset name is ' + str(Dataset) + ', which should be in [IN, PU, PC, SC].')

    shapeor = list(data_hsi.shape)
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])

    if is_pca:
        data_hsi = PCA(n_components=K).fit_transform(data_hsi)
        shapeor[-1] = K
    data = preprocessing.scale(data_hsi)
    shapeor = np.array(shapeor)

    data = data.reshape(shapeor)
    TOTAL_SIZE = np.count_nonzero(gt_hsi)
    CLASSES_NUM = gt_hsi.max()

    return data, gt_hsi, TOTAL_SIZE, CLASSES_NUM


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def select(Dataset, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)

    if Dataset == 'IN':
        amount = [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50]  # IN
    elif Dataset == 'PC':
        amount = [15, 15, 15, 15, 15, 15, 15, 15, 15]  # PC
    elif Dataset == 'PU':
        amount = [200, 200, 200, 200, 200, 200, 200, 200, 200]  # PU
    elif Dataset == 'SC':
        amount = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    elif Dataset == 'HH':
        amount = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    else:
        raise NotImplementedError('Dataset name is ' + str(Dataset) + ', which should be in [IN, PU, PC, SC].')
    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[:nb_val]
        test[i] = indices[nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
