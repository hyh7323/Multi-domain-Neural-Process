from new_module import *


class LatentModel(nn.Module):  # 网络框架，内部模型细节在module文件内
    """
    Latent Model (Attentive Neural Process) 潜在模型
    """

    def __init__(self, num_hidden):  # 训练开头指定model时，定义了num_hidden = 128
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden)
        self.Multidomain_encoder = MultidomainEncoder(num_hidden,num_hidden)
        self.decoder = Decoder(num_hidden)
        self.BCELoss = nn.BCELoss()
        self.MSELoss = nn.MSELoss()

    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)  # 本轮bc目标点数 num_targets = 687

        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)  # 进入my_moudule中的潜在编码器LatentEncoder
        # 截止到这里，上下文数据通过多头注意力模块获得了mu, log_sigma, z,对应着这里的prior_mu, prior_var, prior，这三个都是(12,128)
        # 将context_x, context_y传入潜编码器
        # For training
        if target_y is not None:

            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)  # 与前面相同，将目标点输入潜在编码器内，获取目标点的mu, log_sigma, z
            z = posterior  # 这个z是目标点的隐变量，z=(12,128)

        # For Generation
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H],z变换形状为(12,687,128 )
        r = self.deterministic_encoder(context_x, context_y, target_x)  # [B, T_target, H] r=(12,687,128)
        m = self.Multidomain_encoder(context_x, context_y, target_x)
        # 上一步是将上下文的数据和目标点坐标输入确定性编码器，获得r
        # mu should be the prediction of target y  mu应该是目标y的预测
        y_pred = self.decoder(m, r, z, target_x)  # 将r, z, target_x放入生成解码器g()函数,这里的z是目标点的隐变量
        # 这里的y_pred就是预测结果y_pred=(12,687,1)
        # For Training
        if target_y is not None:
            # get log probability
            b = target_y.shape[0]

            target_y = target_y.view(b, -1, 4, 2)[:, :, 0, :]
            target_y = target_y.squeeze(2)
            bce_loss = self.BCELoss(t.sigmoid(y_pred), target_y)  # 这里是将预测结果与原本目标进行比较计算损失
            mse_loss = self.MSELoss(t.sigmoid(y_pred), target_y)

            # 采用sigmoid激活函数+BCE损失函数，回传的梯度值是正比于预测与真值之差的。bce_loss=1.2018

            # get KL divergence between prior and posterior  获得KL，使用潜编码器得到的上下文输出和目标输出的mu和var,这两个都是(12,128)
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)  # kl = 0.7467

            # maximize prob and minimize KL divergence 最大化概率并最小化 KL 发散
            loss =kl +bce_loss # loss = 1.2018 + 0.7467 =1.9486

        # For Generation
        else:
            log_p = None
            kl = None
            loss = None
        print("loss:"+str(loss)+' '+"mse_loss:"+str(mse_loss)+' '+"kl"+str(kl)+"bce_loss:"+str(bce_loss))
        return y_pred, kl, loss ,mse_loss  # 返回预测值和kl及损失值

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):  #  这里是计算潜编码器的上下文输出与目标输出的KL
        kl_div = (t.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / t.exp(prior_var) - 1. + (
                    prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div