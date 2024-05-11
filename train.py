import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from new_network import LatentModel
from tensorboardX import SummaryWriter
import torch as t
from prepo import collate_fn
import os
import numpy as np
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch import nn
import pywt
from sklearn.metrics import roc_curve, auc
from matplotlib import rc

# Set global font to Times New Roman
rc('font', family='Times New Roman')

path_remot = '/home/hyh/work-space/gear_good/'
# path_local = r'D:/我的坚果云/大论文/数据/good/'
path_409 = '/home/hyh/work-space/gear_test/'
path_in = path_remot

class CNN1d(nn.Module):

  def __init__(self):
    super(CNN1d,self).__init__()
    self.layer1 = nn.Sequential(
          nn.Conv1d(1,100,3,padding=1),
          nn.BatchNorm1d(100),
          nn.ReLU(),
          nn.MaxPool1d(5))
    self.layer2 = nn.Sequential(
          nn.Conv1d(100,50,2),
          nn.BatchNorm1d(50),
          nn.ReLU(),
          nn.MaxPool1d(8))
    self.fc = nn.Linear(19500,1950)
  def forward(self,x):
    #input.shape:(16,1,425)
    x = x.to(t.float32)
    out = self.layer1(x)
    out = out.view(out.size(0),-1)
    out = self.fc(out)
    return out

class CNN_d(nn.Module):

  def __init__(self):
    super(CNN_d,self).__init__()
    self.layer1 = nn.Sequential(
          nn.Conv1d(1,100,3,padding=1),
          nn.BatchNorm1d(100),
          nn.ReLU(),
          nn.MaxPool1d(5))
    self.layer2 = nn.Sequential(
          nn.Conv1d(100,50,2),
          nn.BatchNorm1d(50),
          nn.ReLU(),
          nn.MaxPool1d(8))
    self.fc = nn.Linear(39000,1950)
  def forward(self,x):
    #input.shape:(16,1,425)
    x = x.to(t.float32)
    out = self.layer1(x)
    out = out.view(out.size(0),-1)
    out = self.fc(out)
    return out

Conv = CNN1d()
Conv_d = CNN_d()

class MyDataset_3T(Dataset):
    def __init__(self, mode):
        '''
        data shape: (339, 19500, 14)
        '''
        file_list = os.listdir(path_in)
        file_list.sort(key=lambda x: int(x[5:-4]))
        self.data = []
        # self.label = []
        if mode == 'all':
            path_l7_train = file_list


            for it in tqdm(range(len(path_l7_train)-2)):
                data1 = pd.read_pickle(path_in + path_l7_train[it])
                data2 = pd.read_pickle(path_in + path_l7_train[it+1])
                data3 = pd.read_pickle(path_in + path_l7_train[it + 2])
                data_3t = pd.concat([data1, data2,data3], axis=0, ignore_index=True)#,
                d = t.tensor(data_3t.values)
                # 时域卷积提取特征
                d_trans = t.transpose(d, dim0=0, dim1=1)
                d_trans1 = t.reshape(d_trans, (2, 1, 1950))
                d_Conv = Conv_d(d_trans1)  # total_idx = list[（坐标）,...] len=687,坐标比如（23,0），（0,15）

                # 频域卷积提取特征
                d_numpy = d_trans.numpy()
                cA, cD = pywt.dwt(d_numpy, 'db2')
                cA = t.tensor(cA)
                cA = cA[:, :975]
                cD = t.tensor(cD)
                cD = cD[:, :975]
                cA = t.reshape(cA, (2, 1, 975))
                cD = t.reshape(cD, (2, 1,975))
                cA_conv = Conv(cA)
                cD_conv = Conv(cD)
                d = t.transpose(d,dim0=0,dim1=1)
                data_conv = t.cat((d,d_Conv,cA_conv,cD_conv),dim=0)
                data_conv = t.transpose(data_conv, dim0=0, dim1=1)
                data_conv = data_conv.detach().numpy()
                # c = t.tensor(c)
                self.data.append(data_conv)
            self.data = np.array(self.data)
            self.data = self.data * 0.8 + 0.1
            self.data = self.data.tolist()
            ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MyDataset_bad(Dataset):
    def __init__(self, mode):
        f_list = [r'Inner/',  r'Outer/', r'Roll/']
        path_l1 = os.listdir(path_409 + r'Inner')
        # path_l2 = os.listdir(path_409 + r'Normal')
        path_l2 = os.listdir(path_409 + r'Outer')
        path_l3 = os.listdir(path_409 + r'Roll')
        # path_l5 = os.listdir(path_409 + r'A34_10_01')
        # path_l6 = os.listdir(path_409 + r'A56_11_01')
        path_l7 = os.listdir(path_409 + r'Normal')
        path_l7.sort(key=lambda x: int(x[5:-4]))
        path_l7 = path_l7[:200]
        f_f_list = [path_l1, path_l2, path_l3]
        choice = np.random.choice(np.arange(0, 200 * 3), 80, replace=False)
        self.bad_data = []
        self.data = []
        self.label = []

        if mode == 'try':
            for it in tqdm(choice):
                fd_n = it // 200
                f_n = it % 200
                data = pd.read_pickle(path_409 + f_list[fd_n] + f_f_list[fd_n][f_n])
                data_x = data
                self.bad_data.append(data_x)

            for it in tqdm(range(len(self.bad_data) - 2)):
                data1 = self.bad_data[it]
                data2 = self.bad_data[it + 1]
                data3 = self.bad_data[it + 2]
                data_3t = pd.concat([data1, data2, data3], axis=0, ignore_index=True)
                data_x = data_3t.values
                data = t.tensor(data_x).clone().detach()
                # 时域卷积提取特征
                d_trans = t.transpose(data, dim0=0, dim1=1)
                d_trans1 = t.reshape(d_trans, (2, 1, 1950))
                d_Conv = Conv_d(d_trans1)  # total_idx = list[（坐标）,...] len=687,坐标比如（23,0），（0,15）

                # 频域卷积提取特征
                d_numpy = d_trans.numpy()
                cA, cD = pywt.dwt(d_numpy, 'db2')
                cA = t.tensor(cA).clone().detach()
                cA = cA[:, :975]
                cD = t.tensor(cD).clone().detach()
                cD = cD[:, :975]
                # d_fre = t.cat((cA,cD),dim=-1)[:,:650]
                # d_fretrans = t.reshape(d_fre,(14,1,650))
                cA = t.reshape(cA, (2, 1, 975))
                cD = t.reshape(cD, (2, 1, 975))
                cA_conv = Conv(cA)
                cD_conv = Conv(cD)
                data = t.transpose(data, dim0=0, dim1=1)
                data_conv = t.cat((data, d_Conv, cA_conv, cD_conv), dim=0)
                data_conv = t.transpose(data_conv, dim0=0, dim1=1)
                data_x = data_conv.detach().numpy()
                data_y = 1
                self.data.append(data_x)
                self.label.append(data_y)

            for it in tqdm(range(len(path_l7) - 2)):
                data1 = pd.read_pickle(path_409 + r'Normal/' + path_l7[it])
                data2 = pd.read_pickle(path_409 + r'Normal/' + path_l7[it + 1])
                data3 = pd.read_pickle(path_409 + r'Normal/' + path_l7[it + 2])
                data_3t = pd.concat([data1, data2, data3], axis=0, ignore_index=True)
                data_x = data_3t.values
                data = t.tensor(data_x).clone().detach()
                # 时域卷积提取特征
                d_trans = t.transpose(data, dim0=0, dim1=1)
                d_trans1 = t.reshape(d_trans, (2, 1, 1950))
                d_Conv = Conv_d(d_trans1)  # total_idx = list[（坐标）,...] len=687,坐标比如（23,0），（0,15）

                # 频域卷积提取特征
                d_numpy = d_trans.numpy()
                cA, cD = pywt.dwt(d_numpy, 'db2')
                cA = t.tensor(cA).clone().detach()
                cA = cA[:, :975]
                cD = t.tensor(cD).clone().detach()
                cD = cD[:, :975]
                # d_fre = t.cat((cA,cD),dim=-1)[:,:650]
                # d_fretrans = t.reshape(d_fre,(14,1,650))
                cA = t.reshape(cA, (2, 1, 975))
                cD = t.reshape(cD, (2, 1, 975))
                cA_conv = Conv(cA)
                cD_conv = Conv(cD)
                data = t.transpose(data, dim0=0, dim1=1)
                data_conv = t.cat((data, d_Conv, cA_conv, cD_conv), dim=0)
                data_conv = t.transpose(data_conv, dim0=0, dim1=1)
                data_x = data_conv.detach().numpy()
                data_y = 0
                self.data.append(data_x)
                self.label.append(data_y)

            self.data = np.array(self.data)
            self.data = self.data * 0.8 + 0.1
            self.data = self.data.tolist()
            ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def collate_fn1(batch):  # bc为list[(1950,14),....]len=16

    max_num_context = 1950  # ？
    num_context = np.random.randint(50, 1950)  # extract random number of contexts 提取随机数量上下文，467，产生（10,784）之间的一个随机整数
    num_target = np.random.randint(0, max_num_context - num_context)  # 提取目标数量，这个限制784需要注意，为28*28
    num_total_points = num_context + num_target  # this num should be # of target points 所以这里是抽取上下文和目标的总个数，不满28*28 467 + 220

    # num_total_points = max_num_context
    context_x, context_y, target_x, target_y ,label= list(), list(), list(), list(),list()

    for d,lb in batch:  # batch为[(img,lbble),...],len=12
        d = t.tensor(d)# 将图片pil.image数据转换为tensor,d=(1,28,28)，并将图片的像素值归一化即都除以255


        total_idx = range(0, num_total_points)
        c_idx = total_idx[:num_context]  # 提取上下文的坐标，即Xc，c_idx为上下文坐标list
        c_x, c_y, total_x, total_y = list(), list(), list(), list()
        for idx in c_idx:
            c_y.append(np.array(d[idx, :]))  # 根据坐标提取像素值放入c_y[]中
            c_x.append(t.tensor(idx/num_total_points))  # 将坐标值归一化，即坐标除以27
        for idx in total_idx:
            total_y.append(np.array(d[idx, :]))
            total_x.append(t.tensor(idx/num_total_points))

        # 提取完后，c_x=tensor，(467,2),c_y=Tensor(467,);total_x=(687,2),total_y=(687,)
        c_x, c_y, total_x, total_y = list(map(lambda x: t.FloatTensor(x), (c_x, c_y, total_x, total_y)))  #
        # c_x, total_x = list(map(lambda x: t.FloatTensor(x), (c_x, total_x)))
        context_x.append(c_x)  # context_x=list,[tensor(467,2),...],长度由bc决定
        context_y.append(c_y)  # context_y=list,[tensor(467),...]
        target_x.append(total_x)
        target_y.append(total_y)
        label.append(lb)
    # 下面的stack操作是向量堆叠，即把原本是向量列表的堆叠为高维向量，[tensor(687,)..len=12] --> tensor(12,687,1)
    context_x = t.stack(context_x, dim=0).unsqueeze(-1)
    context_y = t.stack(context_y, dim=0)
    target_x = t.stack(target_x, dim=0).unsqueeze(-1)
    target_y = t.stack(target_y, dim=0)
    # 返回Xc,Yc,Xt,Yt，都是三维的向量，X=tensor(bc,point_num提取的电数,2坐标)，Y=tensor(bc,point_num提取点数，1像素值数)

    return context_x, context_y, target_x, target_y ,label# 这里本次bc的上下文点数为467,目标点数为467 + 220，目标是包含上下文点对的，只是比上下文更多

def adjust_learning_rate(optimizer, step_num):
    lr = 0.001
    for i1 in range(1, step_num):
        if i1 < 1000:
            s = 1
            r = 0.95
        else:
            s = 1.5
            r = 0.9
        if i1 % (50 * s) == 0:
            lr = lr * r
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    lr_c = 0
    train_dataset = MyDataset_3T(mode='all')  # train_dataset=(60000,28,28)
    epochs = 500

    model_single = LatentModel(8 * 16).to(rank)
    model = DDP(model_single, device_ids=[rank],find_unused_parameters=True)
    f1_scores = []
    accuracies = []
    model.train()
    optim = t.optim.Adam(model.parameters(), lr=1e-5)
    writer = SummaryWriter("logs")
    global_step = 0

    dloader = DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True,drop_last=True)#

    # 指向最后一代的模型参数文件
    checkpoint_path = '/data1/hyh/checkpoint_gearMNPforF1/checkpoint_508.pth.tar'

    # 检查文件是否存在
    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))
        # print("Available keys in checkpoint:", checkpoint.keys())  # 打印所有键print("Available keys in checkpoint:", checkpoint.keys())  # 打印所有键
        model.load_state_dict(checkpoint['model'])  # 恢复模型状态
        optim.load_state_dict(checkpoint['optimizer'])  # 恢复优化器状态
    startepoch = 0

    for epoch in range(startepoch,epochs):
        pbar = tqdm(dloader)

        for i, data in enumerate(pbar):  # 从这一步进入dataloader的自定义数据处理函数collate_fn，传入一个bc经过处理后得到data
            global_step += 1
            lr_c = adjust_learning_rate(optim, epoch)

            context_x, context_y, target_x, target_y = data  # data内存放着经过提取的点值(Xc,Yc,Xt,Yt)

            context_x = context_x.to(rank)  # 这里删除了.to(device)
            context_y = context_y.to(rank)
            target_x = target_x.to(rank)
            target_y = target_y.to(rank)
            # pass through the latent model
            y_pred, kl, loss, mse_loss= model(context_x, context_y, target_x,target_y)  # 这里步入network内forward()函数，将数据放入网络，这里删了, loss_MSE
            print("learn_rate=", lr_c)
            # Training step
            optim.zero_grad()  # 梯度置0
            loss.backward()  # 反向传播计算得到每个参数的梯度值
            optim.step()  # 通过梯度下降执行一步参数更新

            if rank == 0:
                # Logging
                writer.add_scalars('gear_training_loss', {

                    'loss': loss,
                    'kl': kl.mean(),'mse_loss':mse_loss
                }, global_step)  # 每训练一个bc都计算一次损失，反向传播更新网络参数，并将损失结果记录到log，global_step每训练一个bc都自增一次，这里删了'MSE': loss_MSE,

        if rank == 1:
            t.save({'model': model.state_dict(),
                    'optimizer': optim.state_dict()},
                   os.path.join(os.getcwd(), '/data1/hyh/checkpoint_gearMNPforF1', 'checkpoint_%d.pth.tar' % (epoch + 1)))

        # if rank == 2:
            # predict and draw
            TN = 0
            TP = 0
            FN = 0
            FP = 0
            mae = list()
            # Initialize tracking lists
            offset = 0.000
            global_step = 0
            MAE_Loss = torch.nn.L1Loss()
            out_list = []
            test_dataset = MyDataset_bad(mode='try')
            dloader_test = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn1, shuffle=False)
            scores = []  # To store predicted scores
            labels = []  # To store true labels
            pbar_test = tqdm(dloader_test)
            for i, data in enumerate(pbar_test):
                    global_step += 1
                    context_x, context_y, target_x, target_y, lable1 = data
                    context_x = context_x.to(rank)  # 这里删除了.to(device)
                    context_y = context_y.to(rank)
                    target_x = target_x.to(rank)
                    target_y = target_y.to(rank)

                    # pass through the latent model
                    y_pred, kl, loss, loss_MSE = model(context_x, context_y, target_x, target_y)

                    b = target_y.shape[0]
                    target_y = target_y.view(b, -1, 4, 2)[:, :, 0, :]
                    target_y = target_y.squeeze(2)
                    mae_1 = MAE_Loss(t.sigmoid(y_pred), target_y)
                    if lable1[0] == 1:
                        mae_1 += offset
                    if lable1[0] == 0:
                        mae.append(mae_1.data.cpu().numpy())
                    if mae_1 <= 0.0445:
                        lable_y = 0
                    else:
                        lable_y = 1

                    if lable1[0] == 0 and lable_y == 0:
                        TP += 1
                    if lable1[0] == 0 and lable_y == 1:
                        FN += 1
                    if lable1[0] == 1 and lable_y == 1:
                        TN += 1
                    if lable1[0] == 1 and lable_y == 0:
                        FP += 1
                    scores.append(mae_1.item())
                    labels.append(lable1[0])
                    out_list.append((mae_1.data.cpu().numpy(), lable1))  # bad:+0.0017

            # FPR = TN / (TP + TN)
            # FNR = FP / (FN + FP)
            # THR = (TP + FN) / (TP + TN + FN + FP)
            mae_std = np.std(mae)
            mae_a = np.mean(mae)
            out_mae = pd.DataFrame(out_list)
            writer = pd.ExcelWriter(r'./good_b_mae.xlsx')
            out_mae.to_excel(writer)
            writer.save()
            out_mae.to_excel('./out_mae.xlsx')
            # Plot ROC curve
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            # 保存FPR和TPR数据
            np.savetxt("DVAE_fpr_data.txt", fpr)
            np.savetxt("DVAE_tpr_data.txt", tpr)
            y_pred = t.sigmoid(y_pred)
            y_pred = y_pred.cpu().detach().numpy()


            # Compute F1 and Accuracy after the epoch
            P = TP / (TP + FP) if (TP + FP) != 0 else 0
            R = TP / (TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
            Accuracy = (TP + TN) / (TP + TN + FN + FP) if (TP + TN + FN + FP) != 0 else 0

            # Store metrics
            f1_scores.append(F1)
            accuracies.append(Accuracy)
        # 将列表转换为 DataFrame
            results = pd.DataFrame({
           'Epoch': range(1, len(f1_scores) + 1),
           'F1_Score': f1_scores,
           'Accuracy': accuracies
       })

        # 写入 CSV 文件
            results.to_csv('training_metrics.csv', index=False)
        # Plotting after all epochs
    # 绘图
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results['Epoch'], results['F1_Score'], marker='o', label='F1 Score')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results['Epoch'], results['Accuracy'], marker='o', label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Optionally, save the figures
    plt.savefig('training_metrics.png', dpi=600)

    torch.save(model, "/data1/hyh/checkpoint_gearMNP/mo1" )
    dist.barrier()


if __name__ == '__main__':
    world_size = 2
    args = None

    # choose local GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["MASTER_ADDR"] = "172.20.13.56"
    os.environ["MASTER_PORT"] = "29500"
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.makedirs('runs', exist_ok=True)

    mp.spawn(main,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
