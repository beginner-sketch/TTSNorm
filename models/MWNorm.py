import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import sys
import numpy as np
from torchsummary import summary

class TNormalize(nn.Module):
    def __init__(self, num_nodes, num_source, channels, track_running_stats=True, momentum=0.1):
        super(TNormalize, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_source, int(num_nodes), 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_source, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_source, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_source, num_nodes, 1))
        self.momentum = momentum
        
        
    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 4), keepdims=True)
            var = x.var((0, 4), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[4] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((4), keepdims=True)
            var = x.var((4), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out

class SNormalize(nn.Module):
    def __init__(self,  channels, num_source):
        super(SNormalize, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, x):
        x_norm = (x - x.mean(3, keepdims=True)) / (x.var(3, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma + self.beta
        return out
    
class Intra_Normalize(nn.Module):
    def __init__(self,  channels, num_source):
        super(Intra_Normalize, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, x):
        x_norm = (x - x.mean((3, 4), keepdims=True)) / (x.var((3, 4), keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma + self.beta
        return out

class Inter_Normalize(nn.Module):
    def __init__(self, channels, num_nodes):
        super(Inter_Normalize, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1,channels,1,num_nodes,1))
        self.gamma = nn.Parameter(torch.ones(1,channels,1,num_nodes,1))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma + self.beta
        return out

class SN_Normalize(nn.Module):
    def __init__(self, channels):
        super(SN_Normalize, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1,1))
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1,1))

    def forward(self, x):
        x_norm = (x - x.mean((2, 3), keepdims=True)) / (x.var((2, 3), keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma + self.beta
        return out
    
class ST_Normalize(nn.Module):
    def __init__(self, channels):
        super(ST_Normalize, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1,1))
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1,1))

    def forward(self, x):
        x_norm = (x - x.mean((2, 4), keepdims=True)) / (x.var((2, 4), keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma + self.beta
        return out

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # A: (s, n, n)
        x = torch.einsum('ncsvl,pvw->ncswl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv3d(c_in, c_out, kernel_size=(1, 1, 1), padding=(0,0,0), stride=(1,1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
                print(1)

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class MWNorm(nn.Module):
    def __init__(self, device, num_nodes, num_source, dropout=0.3, supports=None, gcn_bool=True, intra_bool=False, inter_bool=False,
                 tnorm_bool=False, snorm_bool=False, snnorm_bool=False, stnorm_bool=False, addaptadj=True, aptinit=None, in_dim=4,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(MWNorm, self).__init__()
        self.num_source = num_source
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.in_dim = in_dim
        self.gcn_bool = gcn_bool
        self.intra_bool = intra_bool
        self.inter_bool = inter_bool
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        self.snnorm_bool = snnorm_bool
        self.stnorm_bool = stnorm_bool
        self.addaptadj = addaptadj
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        if self.tnorm_bool:
            self.tnorm = nn.ModuleList()
        if self.snorm_bool:
            self.snorm = nn.ModuleList()
        if self.snnorm_bool:
            self.snnorm = nn.ModuleList()
        if self.intra_bool:
            self.intranorm = nn.ModuleList()
        if self.inter_bool:
            self.internorm = nn.ModuleList()
        if self.stnorm_bool:
            self.stnorm = nn.ModuleList()
        num = int(self.tnorm_bool)+int(self.snorm_bool)+int(self.snnorm_bool)+int(self.intra_bool)+int(self.inter_bool)+1

        self.mlps = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.cross_product = nn.ModuleList()
        self.protos = nn.ParameterList()

        self.start_conv = nn.Conv3d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1,1))
        self.supports = supports

        receptive_field = 1
        self.dropout = nn.Dropout(0.2)
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_source, num_nodes, 128).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(num_source, 128, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        self.dilation = []

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.dilation.append(new_dilation)
                # Normalize
                if self.tnorm_bool:
                    self.tnorm.append(TNormalize(num_nodes, num_source, residual_channels))
                if self.snorm_bool:
                    self.snorm.append(SNormalize(residual_channels, num_source))
                if self.snnorm_bool:
                    self.snnorm.append(SN_Normalize(residual_channels))
                if self.intra_bool:
                    self.intranorm.append(Intra_Normalize(residual_channels, num_source))
                if self.inter_bool:
                    self.internorm.append(Inter_Normalize(residual_channels, num_nodes))
                if self.stnorm_bool:
                    self.stnorm.append(ST_Normalize(residual_channels))
                ###
                self.filter_convs.append(nn.Conv3d(in_channels=num * residual_channels,
                                                   out_channels=num_source * dilation_channels,
                                                   kernel_size=(num_source,1,kernel_size),dilation=(1,1,new_dilation)))

                self.gate_convs.append(nn.Conv3d(in_channels=num * residual_channels,
                                                 out_channels=num_source * dilation_channels,
                                                 kernel_size=(num_source, 1, kernel_size), dilation=(1,1,new_dilation)))
                self.cross_product.append(nn.Bilinear(dilation_channels, dilation_channels, dilation_channels))
                self.mlps.append(nn.Conv3d(in_channels=residual_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(num_source,1, 1)))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv3d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1,1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv3d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1, 1)))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(residual_channels,residual_channels,dropout,support_len=self.supports_len, order=1))


        self.emb1 = nn.Embedding(in_dim, 16)
        self.idx = torch.arange(self.in_dim).to(device)
        self.end_conv_1 = nn.Conv3d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv3d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        input = input.permute(0, 4, 3, 2, 1)
        in_len = input.size(4)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0
        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.matmul(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        # layers
        for i in range(self.blocks * self.layers):
            residual = x
            # Normalize
            x_list = []
            x_list.append(x)
            if self.tnorm_bool:
                x_tnorm = self.tnorm[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.snorm[i](x)
                x_list.append(x_snorm)
            if self.snnorm_bool:
                x_snnorm = self.snnorm[i](x)
                x_list.append(x_snnorm)
            if self.intra_bool:
                x_intra = self.intranorm[i](x)
                x_list.append(x_intra)
            if self.inter_bool:
                x_inter = self.internorm[i](x)
                x_list.append(x_inter)
            if self.stnorm_bool:
                x_stnorm = self.stnorm[i](x)
                x_list.append(x_stnorm)
            x = torch.cat(x_list, dim=1)           
            # dilated convolution
            filter = self.filter_convs[i](x)
            b, _, _, n, t = filter.shape
            filter = torch.tanh(filter).reshape(b, -1, self.num_source, n, t)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate).reshape(b, -1, self.num_source, n, t)
            x = filter * gate
            # parametrized skip connection
            save_x = x
            sk = x
            sk = self.skip_convs[i](sk)
            try:
                skip = skip[:, :, :, :, -sk.size(4):]
            except:
                skip = 0
            skip = sk + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, :, -x.size(4):]
        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(rep)
        return out.permute(0, 1, 3, 2, 4) 

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)
def main():
    n = 98
    batch = 64
    num_source = 4
    channel = 1
    t = 16
    hidden_channels = 16
    n_layers = 4
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '1'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")

    model = MWNorm(device, n, num_source, dropout=0, supports=None, gcn_bool=0, intra_bool=1, inter_bool=1, tnorm_bool=1, snorm_bool=1, snnorm_bool=1,
                    addaptadj=True, aptinit=None, in_dim=1,out_dim=3, residual_channels=hidden_channels, dilation_channels=hidden_channels, 
                    skip_channels=hidden_channels, end_channels=hidden_channels, kernel_size=2, blocks=1, layers=n_layers).to(device)
    summary(model, (t, n, num_source, channel), device=device)
    
if __name__ == '__main__':
    main()



