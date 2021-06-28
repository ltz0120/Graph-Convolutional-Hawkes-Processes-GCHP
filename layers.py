import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # in_feature: the dimension of the input feature vector.
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(max(self.weight.size(1),1)) / 2
        self.weight.data.normal_(0.02, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(0.000002, stdv)

    def forward(self, input, adj):
        # if self.weight.size(1) == 0:
        #     return self.bias
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_transformer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_transformer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # in_feature: the dimension of the input feature vector.
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(max(self.weight.size(1),1)) / 2
        self.weight.data.normal_(0.02, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(0.02, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        print("support:", support.size())   # [2048, 8, 3, 20]
        print("adj:", adj.shape)   # [2048, 8, 3, 3]
        # output = torch.bmm(adj, support)
        output = torch.matmul(adj, support)
        print("output:", output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'