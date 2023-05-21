import argparse
import collections
import math
import time
from sklearn import metrics, preprocessing

import geniter
import record
import torch_optimizer as optim2
import Utils
from torchsummary import summary
import os
from plt_figure import *
from data_loader import load_dataset, select
from ViT_pytorch import GSPFormer_SSI
import spectral as spy


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# # Setting Params


def generate_png_spy(all_iter, net, gt_hsi, device, total_indices, path):
    pred_test = []
    for X, y in all_iter:
        # X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        pred_test.extend(net(X).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    x_label[total_indices] = [i + 1 for i in pred_test]
    x = np.ravel(x_label)
    y_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    gt_re = np.reshape(gt, (gt_hsi.shape[0], gt_hsi.shape[1]))
    spy.save_rgb(path + '.png', y_re, colors=spy.spy_colors)
    spy.save_rgb(path + '_gt.png', gt_re, colors=spy.spy_colors)


parser = argparse.ArgumentParser(description='Pred for HSI')
parser.add_argument(
    '-d', '--dataset', dest='dataset', default='IN', help="Name of dataset.")
parser.add_argument(
    '-e', '--epoch', type=int, dest='epoch', default=250, help="No of epoch")
parser.add_argument(
    '-b', '--batch_size', type=int, dest='batch_size', default=256, help="Batch Size")
parser.add_argument(
    '-p', '--patch', type=int, dest='patch', default=3, help="Length of patch")
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('-m', '--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument(
    '-i', '--iter', type=int, dest='iter', default=0, help="No of iter")
parser.add_argument('-ip', '--is_PCA', type=str2bool, dest='is_PCA', nargs='?',
                    const=True, default=False, help="PCA or not")
args = parser.parse_args()

PARAM_DATASET = args.dataset  # UP,IN,SV, KSC
PATCH_SIZE = args.patch
PARAM_BATCH_SIZE = args.batch_size
PARAM_ITER = args.iter
PARAM_PCA = args.is_PCA
# # Data Loading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ensemble = 1

global Dataset  # UP,IN,SV, KSC
dataset = PARAM_DATASET  # input('Please input the name of Dataset(IN, UP, SV, KSC):')
Dataset = dataset.upper()

# 数据读取
data_hsi, gt_hsi, TOTAL_SIZE, CLASSES_NUM = load_dataset(Dataset, PARAM_PCA)
print(data_hsi.shape)
# one dimension of gt_hsi
gt = gt_hsi.copy()
gt = gt.reshape(-1)
image_x, image_y, BAND = data_hsi.shape
print('The class numbers of the HSI data is:', CLASSES_NUM)
# exit(0)
# 对数据处理
img_rows = PATCH_SIZE
img_cols = PATCH_SIZE
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]

ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]

model = GSPFormer_SSI(image_size=PATCH_SIZE, near_band=args.band_patches, band=BAND, num_classes=CLASSES_NUM, dim=64,
                      depth=5, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode=args.mode).to(device)
summary(model, (1, img_rows, img_cols, BAND))
np.random.seed(seeds[PARAM_ITER])
train_indices, test_indices = select(Dataset, gt)
total_indices = train_indices + test_indices

print('Total Size: %d' % len(total_indices))
print('Train Size: %d' % len(train_indices))
print('Validation Size: %d' % len(train_indices))
print('Test Size: %d' % (len(test_indices)-len(train_indices)))
print('-----Selecting Small Pieces from the Original Cube Data-----')
train_iter, valida_iter, test_iter, all_iter = geniter.generate_iter(train_indices, test_indices, data_hsi,
                                                                         PATCH_SIZE, PARAM_BATCH_SIZE, gt)

# test
full_name_dic = {'IN': 'IndianPines', 'KSC': 'KSC', 'PU': 'PaviaU',
                 'PC': 'PaviaC', 'B': 'Botswana', 'SC': 'Salinas', 'HC': 'HanChuan', 'HH': 'HongHu', 'LK': 'LongKou'}

full_name = full_name_dic[Dataset]

if PARAM_PCA:
    start_path = './PCA'
else:
    start_path = './No_PCA'

dir_PATH = os.path.join(start_path + '/models/', full_name)
dir_PATH = dir_PATH + '/Patch_' + str(PATCH_SIZE)
files = sorted(os.listdir(dir_PATH))

PATH = os.path.join(dir_PATH, files[PARAM_ITER])
print("model path:", PATH)
# exit(0)

model.load_state_dict(torch.load(PATH))
pred_test = []
# real_test = []
tic2 = time.time()
with torch.no_grad():
    for X, y in test_iter:
        X = X.to(device)
        model.eval()
        y_hat = model(X)
        pred_test.extend(np.array(model(X).cpu().argmax(axis=1)))
        # real_test.extend(list(map(int, y.tolist())))
toc2 = time.time()
print(len(pred_test))
# print(real_test)


collections.Counter(pred_test)
gt_test = gt[test_indices] - 1
print(len(gt_test[:-len(train_indices)]))
overall_acc = metrics.accuracy_score(pred_test, gt_test[:-len(train_indices)])
print('overall_acc: ', overall_acc)
confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-len(train_indices)])
print("confusion matrix:\n", confusion_matrix)
each_acc, average_acc = record.aa_and_each_accuracy(confusion_matrix)
kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-len(train_indices)])
print("kappa: ", kappa)

# plot all existed labels
full_name = full_name_dic[Dataset]
save_path = start_path + '/classification_maps/' + full_name + '/Patch_' + str(img_rows)
print('save_path of all existed labels:', save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
Utils.generate_png(
    all_iter, model, gt_hsi, device, total_indices, save_path + '/' + 'iter_' + str(PARAM_ITER))

# plot all pixels
pred_list = get_all_pixels(data_hsi, model, PATCH_SIZE=PATCH_SIZE, device=device, batch_size=256)
print(len(pred_list))
pred_list_color = list_to_colormap_mine(pred_list)
pred_all = np.reshape(pred_list_color, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
classification_map(pred_all, gt_hsi, 300, save_path + '/' + 'iter_' + str(PARAM_ITER) + '_all.png')
print('------Get classification maps successful-------')
