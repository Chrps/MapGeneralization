import torch.nn.functional as F

RGCN_CONFIG = {
    'extra_args': [16, 3, F.relu, 0.1],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

MoNet_CONFIG = {
    'extra_args': [10, [32, 64, 128, 256]],
    'lr': 1e-3,
    'weight_decay': 5e-6,
}

GCN_CONFIG = {
    'extra_args': [8, 8, F.relu, 0.0],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

GAT_CONFIG = {
    'extra_args': [12, 10, [3] * 12 + [1], F.elu, 0.1, 0.1, 0.2, True],
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
    'extra_args': [None, 2, False],
    'lr': 0.2,
    'weight_decay': 5e-6,
}

GIN_CONFIG = {
    'extra_args': [8, 8, 0, True],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}

CHEBNET_CONFIG = {
    'extra_args': [16, 16, 2, True],
    'lr': 1e-3,
    'weight_decay': 5e-4,
}
