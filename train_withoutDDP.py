import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from new_network import LatentModel
from tensorboardX import SummaryWriter
import torchvision
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

# matplotlib.use('TkAgg')

path_remot = '/home/hyh/work-space/goods/'
# path_local = r'D:/我的坚果云/大论文/数据/good/'
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
                data_3t = pd.concat([data1, data2,data3], axis=0, ignore_index=True)
                data_x = data_3t.values[::10, :]

            # for it in tqdm(range(len(path_l7_train) - 2)):
            #     data1 = pd.read_pickle(path_in + path_l7_train[it])
            #     data2 = pd.read_pickle(path_in + path_l7_train[it + 1])
            #     data3 = pd.read_pickle(path_in + path_l7_train[it + 2])
            #     data_3t = pd.concat([data1, data2, data3], axis=0, ignore_index=True)
            #     data_x = data_3t.values[::10, :]

                d = t.tensor(data_x)
                # 时域卷积提取特征
                d_trans = t.transpose(d, dim0=0, dim1=1)
                d_trans1 = t.reshape(d_trans, (14, 1, 1950))
                d_Conv = Conv_d(d_trans1)  # total_idx = list[（坐标）,...] len=687,坐标比如（23,0），（0,15）

                # 频域卷积提取特征
                d_numpy = d_trans.numpy()
                cA, cD = pywt.dwt(d_numpy, 'db2')
                cA = t.tensor(cA)
                cA = cA[:, :975]
                cD = t.tensor(cD)
                cD = cD[:, :975]
                # d_fre = t.cat((cA,cD),dim=-1)[:,:650]
                # d_fretrans = t.reshape(d_fre,(14,1,650))
                cA = t.reshape(cA, (14, 1, 975))
                cD = t.reshape(cD, (14, 1,975))
                cA_conv = Conv(cA)
                cD_conv = Conv(cD)
                d = t.transpose(d,dim0=0,dim1=1)
                data_conv = t.cat((d,d_Conv,cA_conv,cD_conv),dim=0)
                data_conv = t.transpose(data_conv, dim0=0, dim1=1)
                data_conv = data_conv.detach().numpy()
            #     # c = t.tensor(c)

                self.data.append(data_conv)
            self.data = np.array(self.data)
            self.data = self.data * 0.8 + 0.1
            self.data = self.data.tolist()
            ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


'''
train_set = MyDataset_3T(mode='all')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=12, shuffle=True)
'''

'''
def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''


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


def main():
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    lr_c = 0
    train_dataset = MyDataset_3T(mode='all')  # train_dataset=(60000,28,28)
    epochs = 5000
    device = torch.device("cuda:0")
    model = LatentModel(128).to(device)  # 128,隐藏层设定为64 .to(device)

    # model_single = LatentModel(8 * 16).to(rank)
    # model = DDP(model_single, device_ids=[rank])

    model.train()
    optim = t.optim.Adam(model.parameters(), lr=1e-5)
    writer = SummaryWriter("logs")
    global_step = 0

    dloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True,drop_last=True)#

    for epoch in range(epochs):
        pbar = tqdm(dloader)

        for i, data in enumerate(pbar):  # 从这一步进入dataloader的自定义数据处理函数collate_fn，传入一个bc经过处理后得到data
            global_step += 1
            lr_c = adjust_learning_rate(optim, epoch)

            context_x, context_y, target_x, target_y = data  # data内存放着经过提取的点值(Xc,Yc,Xt,Yt)

            # print(context_x.shape)  # context_x = Tensor(12,779,2) context_y = Tensor(12,779,1) (bs,本轮抽取的点数,x/y)
            '''
            context_x = context_x.to(device)  # 这里删除了.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            '''
            context_x = context_x.to(device)  # 这里删除了.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            # pass through the latent model
            y_pred, kl, loss = model(context_x, context_y, target_x,target_y)  # 这里步入network内forward()函数，将数据放入网络，这里删了, loss_MSE
            print("learn_rate=", lr_c)
            # Training step
            optim.zero_grad()  # 梯度置0
            loss.backward()  # 反向传播计算得到每个参数的梯度值
            optim.step()  # 通过梯度下降执行一步参数更新

            # if rank == 0:
                # Logging
            writer.add_scalars('training_loss', {

                'loss': loss,
                 'kl': kl.mean(),
            }, global_step)  # 每训练一个bc都计算一次损失，反向传播更新网络参数，并将损失结果记录到log，global_step每训练一个bc都自增一次，这里删了'MSE': loss_MSE,

        # if rank == 1:
            # pd.DataFrame(y_pred.detach().cpu().numpy()[1, :, :]).to_excel("./result/r317/y_p"+str(epoch)+".xlsx")
            # save model by each epoch
        t.save({'model': model.state_dict(),
                    'optimizer': optim.state_dict()},
                   os.path.join(os.getcwd(), '/home/hyh/checkpoint02', 'checkpoint_%d.pth.tar' % (epoch + 1)))

        # if rank == 2:
            # predict and draw
        file_list = os.listdir(path_in)
        file_list.sort(key=lambda x: int(x[5:-4]))
        id = np.random.choice(range(len(file_list)-2), size=1, replace=False)[0]

        path_l7_train = file_list
        data1 = pd.read_pickle(path_in + path_l7_train[id])
        data2 = pd.read_pickle(path_in + path_l7_train[id+1])
        data3 = pd.read_pickle(path_in + path_l7_train[id + 2])
        data_3t = pd.concat([data1, data2,data3],axis=0, ignore_index=True)

        data_x = data_3t.values[::10, :]
        d = t.tensor(data_x)
            # 时域卷积提取特征
        d_trans = t.transpose(d, dim0=0, dim1=1)
        d_trans1 = t.reshape(d_trans, (14, 1, 1950))
        d_Conv = Conv_d(d_trans1)  # total_idx = list[（坐标）,...] len=687,坐标比如（23,0），（0,15）

            # 频域卷积提取特征
        d_numpy = d_trans.numpy()
        cA, cD = pywt.dwt(d_numpy, 'db2')
        cA = t.tensor(cA)
        cA = cA[:, :975]
        cD = t.tensor(cD)
        cD = cD[:, :975]
            # d_fre = t.cat((cA,cD),dim=-1)[:,:650]
            # d_fretrans = t.reshape(d_fre,(14,1,650))
        cA = t.reshape(cA, (14, 1, 975))
        cD = t.reshape(cD, (14, 1, 975))
        cA_conv = Conv(cA)
        cD_conv = Conv(cD)
        d = t.transpose(d, dim0=0, dim1=1)
        data_conv = t.cat((d, d_Conv, cA_conv, cD_conv), dim=0)
        data_conv = t.transpose(data_conv, dim0=0, dim1=1)
        data_y = data_conv.detach().numpy()
        data_y = np.expand_dims(data_y, axis=0)
        data = data_x.copy()
        data_y = torch.tensor(data_y)

        context_x, context_y, target_x, target_y = collate_fn(data_y)

        context_x = context_x.to(device)  # 这里删除了.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        y_pred, _, _ = model(context_x, context_y, target_x, target_y)
        y_pred = t.sigmoid(y_pred)
        y_pred = y_pred.cpu().detach().numpy()

        context_x = context_x.cpu().numpy()
        context_y = context_y.cpu().numpy()
        target_x = target_x.cpu().numpy()
        target_y = target_y.cpu().numpy()

        x = np.arange(0, 1, 1 / 1950.)
        # b = data.shape[0]
        # data = data.view(b, -1, 4, 14)[:, :, 0, :]
        y = data

        x_scatter = target_x
        y_scatter = target_y

        fig, axs = plt.subplots(7, 2, figsize=(12, 20))
        for i, ax in enumerate(axs.flat):
            if i < 14:
                ax.plot(x, y[:, i], label='ground truth')
                ax.scatter(x_scatter[0, :, 0], y_pred[:, i], s=2, alpha=0.5, c='r', label='predicts')
                ax.legend()
            else:
                    # Remove unused subplots
                fig.delaxes(ax)
        plt.show()
        plt.savefig('./img1/epoch_{:d}.png'.format(epoch))


    torch.save(model, "/home/hyh/checkpoint02/mo1" )
    dist.barrier()


if __name__ == '__main__':
    # world_size = 4
    # args = None
    #
    # # choose local GPUs
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["MASTER_ADDR"] = "172.20.75.105"
    # os.environ["MASTER_PORT"] = "29500"
    # os.makedirs('runs', exist_ok=True)

    main()
