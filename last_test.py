import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from new_network import LatentModel
from tensorboardX import SummaryWriter
import torchvision
import torch as t
from prepo import  collate_fn
import os
import numpy as np
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt

# path_remot = '/home/jiaojian/code_dir/pythonProject35/类别文件_pic/good/'
path_remot = '/home/jiaojian/code_dir/pythonProject35/类别文件_bigpic/good/'

path_in = path_remot


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
            path_l7_train = file_list[:341]
            for it in tqdm(range(len(path_l7_train) - 2)):
                data1 = pd.read_pickle(path_in + path_l7_train[it])
                data2 = pd.read_pickle(path_in + path_l7_train[it + 1])
                data3 = pd.read_pickle(path_in + path_l7_train[it + 2])
                data_3t = pd.concat([data1, data2, data3], axis=0, ignore_index=True)
                # 求均值
                # data_mean10 = list()
                # for i in range(1950):
                #     var = np.mean(data_3t.iloc[10 * i:10 * (i + 1), :].values, axis=0)
                #     data_mean10.append(var)
                # data_x = np.array(data_mean10)
                # pd.DataFrame(data_x).to_excel("./1.xlsx")
                data_x = data_3t.values[::10, :]
                self.data.append(data_x)
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


def adjust_learning_rate(optimizer, num):
    lr = 0.001
    for i in range(num):
        if i % 5 == 0:
            lr = 0.95 * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    lr_c = 0
    train_dataset = MyDataset_3T(mode='all')
    epochs = 300
    model_single = LatentModel(8 * 16).to(rank)
    model = DDP(model_single, device_ids=[rank])

    # checkpoint = t.load('./mydir/checkpoint_4999.pth.tar', map_location='cpu')
    # model.load_state_dict(checkpoint['model'])

    model.train()
    optim = t.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter("logs")
    global_step = 0
    # out_list = []
    dloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
    out_list = list()
    for epoch in range(epochs):
        pbar = tqdm(dloader)
        # if epoch % 5 == 0:
        #   out = pd.DataFrame(out_list)
        #     out.to_excel("./out_list10.xlsx")
        lr_c = adjust_learning_rate(optim, epoch)
        for i, data in enumerate(pbar):  # 从这一步进入dataloader的自定义数据处理函数collate_fn，传入一个bc经过处理后得到data
            global_step += 1

            context_x, context_y, target_x, target_y = data  # data内存放着经过提取的点值(Xc,Yc,Xt,Yt)
            # print(context_x.shape)  # context_x = Tensor(12,779,2) context_y = Tensor(12,779,1) (bs,本轮抽取的点数,x/y)
            '''
            context_x = context_x.to(device)  # 这里删除了.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            '''
            context_x = context_x.to(rank)  # 这里删除了.to(device)
            context_y = context_y.to(rank)
            target_x = target_x.to(rank)
            target_y = target_y.to(rank)
            # pass through the latent model
            y_pred, kl, loss, loss_MSE = model(context_x, context_y, target_x,
                                               target_y)  # 这里步入network内forward()函数，将数据放入网络
            print("learn_rate=", lr_c)
            # out_list.append((y_pred, kl, loss, loss_MSE))

            # Training step
            optim.zero_grad()  # 梯度置0
            loss.backward()  # 反向传播计算得到每个参数的梯度值
            optim.step()  # 通过梯度下降执行一步参数更新
            out_list.append((kl.data.cpu().numpy(), loss.data.cpu().numpy(), loss_MSE.data.cpu().numpy()))
            '''
            if rank == 0:
                # Logging

                writer.add_scalars('training_loss', {
                    'MSE': loss_MSE,
                    'loss': loss,
                    'kl': kl.mean(),
                }, global_step)  # 每训练一个bc都计算一次损失，反向传播更新网络参数，并将损失结果记录到log，global_step每训练一个bc都自增一次

        if rank == 1:
            # pd.DataFrame(y_pred.detach().cpu().numpy()[1, :, :]).to_excel("./result/r317/y_p"+str(epoch)+".xlsx")
            # save model by each epoch
            t.save({'model': model.state_dict(),
                    'optimizer': optim.state_dict()}, os.path.join(os.getcwd(), '/oldhome/jiaojian/checkpoint02', 'checkpoint_%d.pth.tar' % (epoch + 1)))

'''
        if rank == 2:
            # predict and draw
            file_list = os.listdir(path_in)
            file_list.sort(key=lambda x: int(x[5:-4]))
            id = np.random.choice(range(len(file_list) - 3), size=1, replace=False)[0]

            path_l7_train = file_list
            data1 = pd.read_pickle(path_in + path_l7_train[id])
            data2 = pd.read_pickle(path_in + path_l7_train[id + 1])
            data3 = pd.read_pickle(path_in + path_l7_train[id + 2])
            data_3t = pd.concat([data1, data2, data3], axis=0, ignore_index=True)

            data_x = data_3t.values[::10, :]
            data_x = np.expand_dims(data_x, axis=0)
            data = data_x.copy()
            data_x = torch.tensor(data_x)

            context_x, context_y, target_x, target_y = collate_fn(data_x)

            context_x = context_x.to(rank)  # 这里删除了.to(device)
            context_y = context_y.to(rank)
            target_x = target_x.to(rank)
            target_y = target_y.to(rank)
            y_pred, _, _, _ = model(context_x, context_y, target_x, target_y)
            y_pred = t.sigmoid(y_pred)
            y_pred = y_pred.cpu().detach().numpy()

            context_x = context_x.cpu().numpy()
            context_y = context_y.cpu().numpy()
            target_x = target_x.cpu().numpy()
            target_y = target_y.cpu().numpy()

            x = np.arange(0, 1, 1 / 1950.)
            y = data

            x_scatter = target_x
            y_scatter = target_y

            fig, axs = plt.subplots(7, 2, figsize=(15, 20))
            for i, ax in enumerate(axs.flat):
                if i < 14:
                    ax.plot(x, y[0, :, i], label='ground truth')
                    ax.scatter(x_scatter[0, :, 0], y_pred[0, :, i], s=2, alpha=0.5, c='r', label='predicts')
                    ax.legend()
                else:
                    # Remove unused subplots
                    fig.delaxes(ax)

            plt.savefig('./img/epoch_{:d}.png'.format(epoch))

        dist.barrier()
    out_mae = pd.DataFrame(out_list)
    writer = pd.ExcelWriter(r'./kl_loss_mse.xlsx')
    out_mae.to_excel(writer)
    writer.save()


if __name__ == '__main__':
    world_size = 2
    args = None

    # choose local GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    os.environ["MASTER_ADDR"] = "172.20.75.105"
    os.environ["MASTER_PORT"] = "29500"
    os.makedirs('runs', exist_ok=True)

    mp.spawn(main,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

