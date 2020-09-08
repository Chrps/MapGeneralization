import torch.nn as nn
import torch
import dgl.function as fn
from functools import partial
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv, GINConv,\
    APPNPConv, TAGConv, SGConv, AGNNConv, ChebConv, GMMConv
from src.configs import GCN_CONFIG, GAT_CONFIG, GRAPHSAGE_CONFIG, \
    APPNP_CONFIG, TAGCN_CONFIG, AGNN_CONFIG, SGC_CONFIG, GIN_CONFIG, CHEBNET_CONFIG, MoNet_CONFIG
from dgl.nn.pytorch.glob import MaxPooling

def get_model_and_config(name):
    name = name.lower()
    if name == 'gcn':
        return GCN, GCN_CONFIG
    elif name == 'gat':
        return GAT, GAT_CONFIG
    elif name == 'monet':
        return MoNet, MoNet_CONFIG
    elif name == 'graphsage':
        return GraphSAGE, GRAPHSAGE_CONFIG
    elif name == 'appnp':
        return APPNP, APPNP_CONFIG
    elif name == 'tagcn':
        return TAGCN, TAGCN_CONFIG
    elif name == 'agnn':
        return AGNN, AGNN_CONFIG
    elif name == 'sgc':
        return SGC, SGC_CONFIG
    elif name == 'gin':
        return GIN, GIN_CONFIG
    elif name == 'chebnet':
        return ChebNet, CHEBNET_CONFIG


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 size_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, size_hidden, activation=activation, allow_zero_in_degree=False))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(size_hidden, size_hidden, activation=activation, allow_zero_in_degree=False))
        # output layer
        self.layers.append(GraphConv(size_hidden, n_classes, allow_zero_in_degree=False))
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, g, features):

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 num_classes,
                 size_hidden,
                 n_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_feats, size_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=False))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = size_hidden * num_heads
            self.gat_layers.append(GATConv(
                size_hidden * heads[l-1], size_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=False))
        # output projection
        self.gat_layers.append(GATConv(
            size_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, allow_zero_in_degree=False))

    def forward(self, g, features):
        h = features
        for l in range(self.n_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 size_hidden,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, size_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(size_hidden, size_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(size_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class APPNP(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 size_hidden,
                 n_layers,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, size_hidden))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(size_hidden, size_hidden))
        # output layer
        self.layers.append(nn.Linear(size_hidden, n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h


class TAGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 size_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(TAGCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_feats, size_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(TAGConv(size_hidden, size_hidden, activation=activation))
        # output layer
        self.layers.append(TAGConv(size_hidden, n_classes)) #activation=None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class AGNN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 size_hidden,
                 n_layers,
                 init_beta,
                 learn_beta,
                 dropout):
        super(AGNN, self).__init__()
        self.layers = nn.ModuleList(
            [AGNNConv(init_beta, learn_beta, allow_zero_in_degree=False) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, size_hidden),
            nn.ReLU()
        )
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(size_hidden, n_classes)
        )

    def forward(self, g, features):
        h = self.proj(features)
        for layer in self.layers:
            h = layer(g, h)
        return self.cls(h)

class SGC(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 k,
                 bias):
        super(SGC, self).__init__()
        self.net = SGConv(in_feats,
                          n_classes,
                          k=k,
                          cached=False,
                          bias=bias,
                          norm=None)

    def forward(self, g, features):
        return self.net(g, features)


class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 size_hidden,
                 n_layers,
                 init_eps,
                 learn_eps):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Dropout(0.0),
                    nn.Linear(in_feats, size_hidden),
                    nn.ReLU(),
                ),
                'mean',
                init_eps,
                learn_eps
            )
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Dropout(0.0),
                        nn.Linear(size_hidden, size_hidden),
                        nn.ReLU()
                    ),
                    'mean',
                    init_eps,
                    learn_eps
                )
            )
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Dropout(0.0),
                    nn.Linear(size_hidden, n_classes),
                ),
                'mean',
                init_eps,
                learn_eps
            )
        )

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class ChebNet(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 size_hidden,
                 n_layers,
                 k,
                 activation,
                 bias):
        super(ChebNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            ChebConv(in_feats, size_hidden, k, activation, bias)
        )
        for _ in range(n_layers - 1):
            self.layers.append(
                ChebConv(size_hidden, size_hidden, k, activation, bias)
            )

        self.layers.append(
            ChebConv(size_hidden, n_classes, k, activation, bias)
        )

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

#TODO: forward() missing 1 required positional argument: 'pseudo'
class MoNet(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 pseudo_coord,
                 n_kernels,
                 hiddens):
        super(MoNet, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(
            GMMConv(in_feats, hiddens[0], dim=pseudo_coord, n_kernels=n_kernels))

        # Hidden layer
        for i in range(1, len(hiddens)):
            self.layers.append(GMMConv(hiddens[i - 1], hiddens[i], dim=pseudo_coord, n_kernels=n_kernels))

        self.cls = nn.Sequential(
            nn.Linear(hiddens[-1], n_classes),
            nn.LogSoftmax()
        )

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        return h
