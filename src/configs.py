import torch.nn.functional as F

RGCN_CONFIG = {
    'extra_args': [16, 3, F.relu, 0.1],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

#MoNet_CONFIG = {
#    'extra_args': [10, [32, 64, 128, 256]],
#    'lr': 1e-3,
#    'weight_decay': 5e-6,
#}

MoNet_CONFIG = {
    'extra_args': [10, 10, [32, 64, 128, 256]],
    'lr': 1e-3,
    'weight_decay': 5e-6,
}

GCN_CONFIG = {
    'extra_args': [8, 8, F.relu, 0.0],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

GAT_CONFIG = {
    'extra_args': [20, 10, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], F.elu, 0.0, 0.0, 0.2, True], # old:  [8, 8, [3] * 8 + [1], F.elu, 0.0, 0.0, 0.2, True], NOTE: Model Config: [20, 10, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], <function elu at 0x0000024166DF5268>, 0.0, 0.0, 0.2, True] lr: 0.001 weight decay: 0.0005
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

GRAPHSAGE_CONFIG = {
    'extra_args': [12, 8, F.relu, 0.0, 'gcn'],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

APPNP_CONFIG = {
    'extra_args': [8, 4, F.relu, 0.0, 0.0, 0.1, 3],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

TAGCN_CONFIG = {
    'extra_args': [8, 6, F.relu, 0.0],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

AGNN_CONFIG = {
    'extra_args': [8, 8, 1.0, True, 0.0],
    'lr': 1e-2,
    'weight_decay': 5e-4,
}

SGC_CONFIG = {
    'extra_args': [2, True],
    'lr': 1e-2,
    'weight_decay': 5e-4,
}

GIN_CONFIG = {
    'extra_args': [8, 8, 0, True],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

CHEBNET_CONFIG = {
    'extra_args': [8, 8, 1, F.relu, True],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}
