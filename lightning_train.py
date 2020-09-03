import argparse
import os
import torch.nn.functional as F

import torch
import pytorch_lightning as pl


hparams = {'data_path': 'data/Public',
           'train_list': 'test_list.txt',
           'test_list': 'test_list.txt',
           'val_list': 'valid_list.txt',
           'network': 'gat', #gcn', 'gat', 'monet', 'graphsage', 'appnp', 'tagcn', 'agnn', 'sgc', 'gin', 'chebnet'
           'batch_size': 7,
           'n_classes': 2,
           'n_features': 5,
           'windowing': False,
           'n_workers':4,
           'gpus':None}

if __name__ == '__main__':
    """
    Trains a patch classifier.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fine", type=str,
                    default='fine', help="Flag")

    args = vars(ap.parse_args())

    logger = pl.loggers.TensorBoardLogger('lightning_logs/cpu-net-{}_bs-{}'.format(hparams['network'],hparams['batch_size']))
    trainer = pl.Trainer(gpus=hparams['gpus'],
                         max_epochs=1000,
                         logger=logger,
                         default_save_path='trained_models/')

    from lightning_model import LightningNodeClassifier
    model = LightningNodeClassifier(hparams)

    trainer.fit(model)
    #torch.save(model.model, "trained_models/model.pt")

    #trainer.test(model)
