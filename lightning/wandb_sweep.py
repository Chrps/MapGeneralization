import argparse
import os

import wandb
os.environ["WANDB_API_KEY"] = '0ea265928f97057d48eca8a9501e9a078402ccc1' #'ef5a53f78faf2e2ebc938760e84e3b7ad38fb28c'


if __name__ == '__main__':
    """
    Trains a patch classifier.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--command", type=str,
                    default='wandb agent markpp/MapGeneralization/dbxr916q', help="wandb agent command")

    args = vars(ap.parse_args())

    wandb.login()
    os.system(args['command'])
