import torch as t
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """

    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        # ba = y.shape[0]
        # if ba != 2:
        #     ...
        b = y.size(0)
        # # concat location (x) and value (y)
        y_d = y.view(b,-1,4,2)[:,:,0,:]

        encoder_input = t.cat([x, y_d], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = t.relu(self.penultimate_layer(hidden))

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        # reparameterization trick  randn_like返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
        std = t.exp(0.5 * log_sigma)
        eps = t.randn_like(std)
        z = eps.mul(std).add_(mu)

        # return distribution
        return mu, log_sigma, z


class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(1, num_hidden)
        self.target_projection = Linear(1, num_hidden)

    def forward(self, context_x, context_y, target_x):
        b = context_y.size(0)

        # concat context location (x), context value (y)
        y_d = context_y.view(b, -1, 4, 2)[:, :, 0, :]

        encoder_input = t.cat([context_x,y_d], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

            # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)
        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query


class MultidomainEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(MultidomainEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([SCAttention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(1, num_hidden)
        self.target_projection = Linear(1, num_hidden)

    def forward(self, context_x, context_y, target_x):
        b = context_x.size(0)
        # concat context location (x), context value (y)

        y_d = context_y.view(b, -1, 4, 2)[:, :, 0, :]
        y_dconv = context_y.view(b, -1, 4, 2)[:, :, 1, :]
        y_cA = context_y.view(b, -1, 4, 2)[:, :, 2, :]
        y_cD = context_y.view(b, -1, 4, 2)[:, :, 3, :]
        # encoder_input = t.cat([context_x, context_y], dim=-1)

        encoder_input = t.cat([context_x,y_d], dim=-1)
        encoder_input1 = t.cat([context_x, y_dconv], dim=-1)
        encoder_input2 = t.cat([context_x, y_cA], dim=-1)
        encoder_input3 = t.cat([context_x, y_cD], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)
        encoder_input1 = self.input_projection(encoder_input1)
        encoder_input2 = self.input_projection(encoder_input2)
        encoder_input3 = self.input_projection(encoder_input3)


        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input,encoder_input, encoder_input, encoder_input)

        # multi_domain fuse layer 1
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input, encoder_input1,encoder_input1,
                                             encoder_input1)
        # multi_domain fuse layer 2
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input, encoder_input2,encoder_input2,
                                             encoder_input2)
        # multi_domain fuse layer 3
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input, encoder_input3,encoder_input3,
                                             encoder_input3)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query

class Decoder(nn.Module):
    """
    Decoder for generation
    """

    def __init__(self, num_hidden):
        super(Decoder, self).__init__()
        self.target_projection = Linear(1, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 4 , num_hidden* 4 , w_init='relu') for _ in range(3)])
        self.lstm = nn.ModuleList([lstm_decoder(num_hidden * 4, num_hidden*4) for _ in range(3)])
        self.final_projection = Linear(num_hidden * 4, 2)

    def forward(self, m, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,target_x)
        hidden = t.cat([t.cat([t.cat([r, z], dim=-1),m],dim=-1), target_x], dim=-1)

        # hidden = t.cat([t.cat([r, z], dim=-1), target_x], dim=-1)

        # # lstm layers
        # for linear in self.lstm:
        #     hidden, _ = linear(hidden)

        # mlp layers
        for linear in self.linears:
            hidden = t.relu(linear(hidden))

        # get mu and sigma
        y_pred = self.final_projection(hidden)


        return y_pred


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = t.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = t.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # 数据格式（batchsize，数据点数，隐藏层的个数）
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)  # 除了给定的维度，其余按最小粒度合成
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = t.cat([residual, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class SCAttention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=2):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(SCAttention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query, key2, value2, query2):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make  self multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # 数据格式（batchsize，数据点数，隐藏层的个数）
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        skey = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)  # 除了给定的维度，其余按最小粒度合成
        svalue = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        squery = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Make cross multihead
        key = self.key(key2).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # 数据格式（batchsize，数据点数，隐藏层的个数）
        value = self.value(value2).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query2).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        ckey = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)  # 除了给定的维度，其余按最小粒度合成
        cvalue = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        cquery = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)
        # Get context vector

        s_result, s_attns = self.multihead(skey, svalue, squery)

        c_result, c_attns = self.multihead(ckey, cvalue, cquery)

        sc_result, sc_attns = self.multihead(skey, svalue, cquery)

        cs_result, cs_attns = self.multihead(ckey, cvalue, squery)

        # 自交叉融合
        result = s_result + c_result + sc_result + cs_result
        attns = s_attns + c_attns + sc_attns + cs_attns

        # result = t.cat([s_result,c_result,sc_result,cs_result],dim=-1)
        # attns = t.cat([s_attns , c_attns, sc_attns , cs_attns],dim=-1)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = t.cat([residual, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=2):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        lstm_out, self.hidden = self.lstm(x_input)
        output = self.linear(lstm_out)

        return output, self.hidden

