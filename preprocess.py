import torch as t
import numpy as np





def collate_fn(batch):  # bc为list[(1950,14),....]len=16
    # Puts each data field into a tensor with outer dimension batch size
    # assert isinstance(batch[0], tuple)  # batch的格式为list长度为bc长度，[(28*28图片，标签)，....]，这里断言检查数据是否为元组

    # trans = torchvision.transforms.ToTensor()  # 将 PIL Image 或 numpy.ndarray 转为 tensor
    # batch_size = len(batch)

    max_num_context = 1950  # ？
    num_context = np.random.randint(50, 1950)  # extract random number of contexts 提取随机数量上下文，467，产生（10,784）之间的一个随机整数
    num_target = np.random.randint(0, max_num_context - num_context)  # 提取目标数量，这个限制784需要注意，为28*28
    num_total_points = num_context + num_target  # this num should be # of target points 所以这里是抽取上下文和目标的总个数，不满28*28 467 + 220

    # num_context = 1300
    # num_target = 650
    # num_total_points = num_context + num_target

    # num_total_points = max_num_context
    context_x, context_y, target_x, target_y = list(), list(), list(), list()

    for d in batch:  # batch为[(img,lbble),...],len=12
        d = t.tensor(d) # 将图片pil.image数据转换为tensor,d=(1,28,28)，并将图片的像素值归一化即都除以255


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

    # 此刻完成了一个batch的数据提取操作，得到了上下文数据和目标数据，target_y=list，[tensor(687,)..len=12],target_x=list,[tensor(687,2)]
    # context_x=tensor(12,467,2),context_y=tensor(12,467,1)
    # 下面的stack操作是向量堆叠，即把原本是向量列表的堆叠为高维向量，[tensor(687,)..len=12] --> tensor(12,687,1)
    context_x = t.stack(context_x, dim=0).unsqueeze(-1)
    context_y = t.stack(context_y, dim=0)
    target_x = t.stack(target_x, dim=0).unsqueeze(-1)
    target_y = t.stack(target_y, dim=0)
    # 返回Xc,Yc,Xt,Yt，都是三维的向量，X=tensor(bc,point_num提取的电数,2坐标)，Y=tensor(bc,point_num提取点数，1像素值数)

    return context_x, context_y, target_x, target_y # 这里本次bc的上下文点数为467,目标点数为467 + 220，目标是包含上下文点对的，只是比上下文更多
