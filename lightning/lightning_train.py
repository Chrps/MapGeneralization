import argparse
import os
import torch.nn.functional as F

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
os.environ["WANDB_API_KEY"] = 'ef5a53f78faf2e2ebc938760e84e3b7ad38fb28c'

from lightning_callbacks import chpt_callback


if __name__ == '__main__':
    """
    Trains grpah node classifier.

    Command:
        python lightning_train.py
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-wb", "--wandb", type=bool,
                    default=True, help="Use wandb?")
    ap.add_argument("-net", "--network", type=str,
                    default='gat', help="Network architecture")
    ap.add_argument("-bs", "--batch_size", type=int,
                    default=7, help="Batch size")
    ap.add_argument("-ls", "--learning_rate", type=float,
                    default=0.001, help="Learning rate")
    ap.add_argument("-sh", "--size_hidden", type=int,
                    default=8, help="Width of hidden layer")
    ap.add_argument("-nl", "--n_layers", type=int,
                    default=8, help="Number of layers")
    ap.add_argument("-win", "--windowing", type=str,
                    default="false", help="Windowing")
    ap.add_argument("-s", "--save", type=bool,
                    default=False, help="Save model")
    args = vars(ap.parse_args())

    args['windowing'] = args['windowing'].lower() in ("yes", "true", "t", "1")

    # broken
    # chebnet('bool' object is not callable, chebconv.py", line 159)
    # sgc(Expected input batch_size (12782) to match target batch_size (23825) line 2113, in nll_loss)
    # monet(forward() missing 1 required positional argument: 'pseudo', )
    hparams_defaults = {
      'data_path': '../data/Public',
      'train_list': 'train_list.txt',
      'test_list': 'test_list.txt',
      'val_list': 'valid_list.txt',
      'network': args['network'], #gcn', 'gat', '', 'graphsage', 'appnp', 'tagcn', 'agnn', '', 'gin', ''
      'batch_size': args['batch_size'],
      'learning_rate': args['learning_rate'],
      'n_classes': 2,
      'n_features': 5,
      'n_epochs': 250,
      'size_hidden': args['size_hidden'],
      'n_layers': args['n_layers'],
      'windowing': args['windowing'],
      'n_workers':6,
      'gpus':None
     }

    hparams = hparams_defaults
    exp_name = 'net-{}_bs-{}'.format(hparams['network'],hparams['batch_size'])

    if args['wandb']:
        wandb_logger = WandbLogger(project='MapGeneralization')
        #wandb.log_hyperparams(hparams)
        trainer = pl.Trainer(gpus=hparams['gpus'],
                             max_epochs=hparams['n_epochs'],
                             logger=wandb_logger,
                             default_root_dir='trained_models/',
                             checkpoint_callback=None) #chpt_callback
    else:
        logger = pl.loggers.TensorBoardLogger('lightning_logs/{}'.format(exp_name))
        trainer = pl.Trainer(gpus=hparams['gpus'],
                             max_epochs=hparams['n_epochs'],
                             logger=logger,
                             default_root_dir='trained_models/',
                             checkpoint_callback=None) #chpt_callback

    from lightning_model import LightningNodeClassifier
    model = LightningNodeClassifier(hparams)

    if args['wandb']:
        print("for inspection of gradients")
        #wandb_logger.watch(model, log_freq=10)

    trainer.fit(model)

    if args['save']:
        if not os.path.exists('trained_models/'):
            os.mkdir('trained_models/')
        # save model
        output_dir = os.path.join('trained_models/',exp_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        model_path = os.path.join(output_dir,'model_dict.pth')
        torch.save(model.model, os.path.join(output_dir,'model.pt'))
        #torch.save(model.model.state_dict(), model_path)


    trainer.test(model, test_dataloaders=model.test_dataloader())

    #TEST RESULTS
    #{'test_class0_acc': 0.8563412300303268,
    # 'test_class1_acc': 0.8564249937985233,
    # 'test_loss': array(0.36615205, dtype=float32),
    # 'test_overall_acc': 0.8563831119144251}

    #trainer.test(model, test_dataloaders=model.val_dataloader())
    #print(res)
