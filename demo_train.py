import argparse
import collections
import time
import numpy as np
import torch
from torchsummary import summary
import torch.optim as optim
from sklearn import metrics
import geniter
import torch.backends.cudnn as cudnn
import torch_optimizer as optim2
import os
from data_loader import load_dataset, select
from ViT_pytorch import GSPFormer_SSI
import record


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# # Setting Params
parser = argparse.ArgumentParser(description='Training for HSI')
parser.add_argument(
    '-d', '--dataset', dest='dataset', choices=['IN', 'PC', 'PU', 'SC', 'HH'], default='IN', help="Name of dataset.")
parser.add_argument('-m', '--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--seed', type=int, default=0, help='number of seed')

parser.add_argument('-b', '--batch_size', type=int, default=64, help='number of batch size')

parser.add_argument('-p', '--patches', type=int, default=3, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('-e', '--epochs', type=int, default=250, help='epoch number')
parser.add_argument('-i', '--iter', type=int, dest='iter', default=10, help="No of iter")
parser.add_argument('-ip', '--is_PCA', type=str2bool, dest='is_PCA', nargs='?',
                    const=True, default=False, help="PCA or not")

parser.add_argument('-l', '--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
parser.add_argument('-o', '--optimizer', dest='optimizer', default='adam', help="Name of optimizer.")
parser.add_argument('--early_stop', type=bool, dest='early_stop', default=False, help="early stop or not")
parser.add_argument(
    '-en', '--early_num', type=int, dest='early_num', default=20, help="Early Num")
args = parser.parse_args()


def train(index_iter, net, train_iter, valida_iter, loss, optimizer, scheduler,
          device, epochs, loss_val_save_path, model_save_path, early_stopping=True, early_num=50):
    loss_list = [100]
    early_epoch = 0
    net = net.to(device)
    net.train()

    print("training on ", device)
    start = time.time()

    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    acc_max = 0

    for epoch in range(epochs):
        # scheduler.step()
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        # lr_adjust.step()
        scheduler.step()
        valida_acc, valida_loss = record.evaluate_accuracy(
            valida_iter, net, loss, device)
        loss_list.append(valida_loss)

        train_loss_list.append(train_l_sum)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        PATH = "./net_DBA.pt"
        if early_stopping and loss_list[-2] < loss_list[-1]:
            if early_epoch == 0:  # and valida_acc > 0.9:
                torch.save(net.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0

        if valida_acc_list[-1] >= acc_max:
            acc_max = valida_acc_list[-1]
            torch.save(net.state_dict(),
                       model_save_path + "/" + str(net.name)
                       + '_' + str(index_iter) + '_' + 'best' + '.pt')
            best_epoch = epoch
        print(
            'epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, lr %.6f, '
            'time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               valida_loss, valida_acc, optimizer.state_dict()['param_groups'][0]['lr'], time.time() - time_epoch))

    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec, best epoch %d.'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
             time.time() - start, best_epoch+1))


def main():
    PARAM_DATASET = args.dataset  # UP,IN,SV, KSC
    PARAM_EPOCH = args.epochs
    PARAM_ITER = args.iter
    PATCH_SIZE = args.patches
    PARAM_PCA = args.is_PCA

    PARAM_OPTIM = args.optimizer
    PARAM_LR = args.lr
    PARAM_BATCH_SIZE = args.batch_size
    EARLY_NUM = args.early_num

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for Monte Carlo runs
    # seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ensemble = 1

    global Dataset  # UP,IN,SV, KSC
    dataset = PARAM_DATASET  # input('Please input the name of Dataset(IN, UP, SV, KSC):')
    Dataset = dataset.upper()
    data_hsi, gt_hsi, TOTAL_SIZE, CLASSES_NUM = load_dataset(Dataset, PARAM_PCA)

    print('The class numbers of the HSI data is:', CLASSES_NUM)
    print('The input hyperspectral data shape after PCA is:', data_hsi.shape)

    # one dimension of gt_hsi
    gt = gt_hsi.copy()
    gt = gt.reshape(-1)

    # -----Importing Setting Parameters-----
    BAND = data_hsi.shape[2]
    loss = torch.nn.CrossEntropyLoss()

    model = GSPFormer_SSI(image_size=PATCH_SIZE, near_band=args.band_patches, band=BAND, num_classes=CLASSES_NUM,
                          dim=64,
                          depth=5, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode=args.mode).cuda()
    summary(model, (1, PATCH_SIZE, PATCH_SIZE, BAND))
    # exit(0)

    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((PARAM_ITER, CLASSES_NUM))

    # save path
    full_name_dic = {'IN': 'IndianPines', 'KSC': 'KSC', 'PU': 'PaviaU',
                     'PC': 'PaviaC', 'B': 'Botswana', 'SC': 'Salinas', 'HC': 'HanChuan', 'HH': 'HongHu',
                     'LK': 'LongKou'}
    full_name = full_name_dic[Dataset]

    if PARAM_PCA:
        start_path = './PCA'
    else:
        start_path = './No_PCA'

    model_save_path = start_path + '/models/' + full_name + '/Patch_' + str(PATCH_SIZE)
    report_save_path = start_path + '/report/' + full_name + '/txt'
    loss_val_save_path = start_path + '/report/' + full_name + '/excel' + '/Patch_' + str(PATCH_SIZE)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(report_save_path):
        os.makedirs(report_save_path)

    cudnn.deterministic = True
    cudnn.benchmark = False
    for index_iter in range(PARAM_ITER):
        print('iter %d/%d' % (index_iter + 1, PARAM_ITER))
        # define the model
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        net = GSPFormer_SSI(image_size=PATCH_SIZE, near_band=args.band_patches, band=BAND, num_classes=CLASSES_NUM,
                            dim=64,
                            depth=5, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode=args.mode).cuda()

        if PARAM_OPTIM == 'diffgrad':
            optimizer = optim2.DiffGrad(
                net.parameters(),
                lr=PARAM_LR,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=args.weight_decay)  # weight_decay=0.0001)
        elif PARAM_OPTIM == 'adam':
            optimizer = optim.Adam(
                net.parameters(),
                lr=PARAM_LR,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=args.weight_decay)
        elif PARAM_OPTIM == 'sgd':
            optimizer = optim.SGD(
                net.parameters(),
                lr=PARAM_LR,
                momentum=0.9,
                weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('Optimizer is '
                                      + str(PARAM_OPTIM) + ', which should be in [adam, sgd, diffgrad].')

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 10, gamma=args.gamma)

        np.random.seed(seeds[index_iter])
        # use gt to split dataset
        train_indices, test_indices = select(Dataset, gt)
        total_indices = train_indices + test_indices

        print('Total Size: %d' % len(total_indices))
        print('Train Size: %d' % len(train_indices))
        print('Validation Size: %d' % len(train_indices))
        print('Test Size: %d' % (len(test_indices) - len(train_indices)))

        print('-----Selecting Small Pieces from the Original Cube Data-----')
        train_iter, valida_iter, test_iter, all_iter = geniter.generate_iter(train_indices, test_indices, data_hsi,
                                                                             PATCH_SIZE, PARAM_BATCH_SIZE, gt)
        tic1 = time.time()
        train(index_iter, net, train_iter, valida_iter, loss, optimizer, scheduler, device, epochs=PARAM_EPOCH,
              loss_val_save_path=loss_val_save_path, model_save_path=model_save_path,
              early_stopping=args.early_stop, early_num=EARLY_NUM)
        toc1 = time.time()

        pred_test = []
        net.eval()
        net.load_state_dict(torch.load(model_save_path + "/" + str(net.name)
                                       + '_' + str(index_iter) + '_' + 'best' + '.pt'))
        tic2 = time.time()
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(device)
                pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
        toc2 = time.time()
        collections.Counter(pred_test)
        gt_test = gt[test_indices] - 1

        overall_acc = metrics.accuracy_score(pred_test, gt_test[:-len(train_indices)])
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-len(train_indices)])
        each_acc, average_acc = record.aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-len(train_indices)])

        # torch.save(net.state_dict(),
        #            model_save_path + "/" + str(net.name) + '_' + str(index_iter) + '_' + str(
        #                round(overall_acc, 3)) + '.pt')
        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[index_iter, :] = each_acc

    # # Map, Records
    print("--------" + " Training Finished-----------")
    record.record_output(
        OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
        report_save_path + '/' + str(model.name) + '_patch_' + str(PATCH_SIZE) + '_' + Dataset +
        '_lr_' + str(PARAM_LR) + PARAM_OPTIM + '.txt')


if __name__ == '__main__':
    main()