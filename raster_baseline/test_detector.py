import argparse
import os

import torch
import pytorch_lightning as pl

import config

from data_loader import prepare_data_from_list, standard_dataloader

import cv2

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--top", type=str,
                    default='trained_models/model_dict.pth', help="Path to trained model")
    ap.add_argument("-m", "--middle", type=str,
                    default='trained_models/model_dict_middle.pth', help="Path to trained model")
    ap.add_argument("-l", "--left", type=str,
                    default='trained_models/model_dict_left.pth', help="Path to trained model")
    ap.add_argument("-r", "--right", type=str,
                    default='trained_models/model_dict_right.pth', help="Path to trained model")
    args = vars(ap.parse_args())


    from lightning_model import LightningDetector
    model = LightningDetector(config.hparams)
    model.model.load_state_dict(torch.load(args["top"]))
    model.eval()
    '''
    model_m = LightningDetector(config.hparams)
    model_m.model.load_state_dict(torch.load(args["middle"]))
    model_m.eval()


    model_l = LightningDetector(config.hparams)
    model_l.model.load_state_dict(torch.load(args["left"]))
    model_l.eval()

    model_r = LightningDetector(config.hparams)
    model_r.model.load_state_dict(torch.load(args["right"]))
    model_r.eval()
    '''
    data_val = prepare_data_from_list(os.path.join(config.hparams.data_path,config.hparams.train_list),
                                      crop_size=config.hparams.input_size)

    count = 0
    for sample in data_val:
        input, gt, img, offset = sample
        c, w, h = input.shape
        #print(gt)
        count = count + 1

        #input[0] = input[0]*0.229 + 0.485
        #input[1] = input[1]*0.224 + 0.456
        #input[2] = input[2]*0.225 + 0.406
        res = input.mul(255).permute(1, 2, 0).byte().numpy()
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        top_x, top_y = int(gt[0]*w), int(gt[1]*h)
        res = cv2.circle(res, (top_x, top_y), 2, (0,255,0), 5)

        m = gt[:2] - gt[2:] * 0.5
        m_x_r, m_y_r = int(m[0]*w), int(m[1]*h)
        res = cv2.line(res, (top_x, top_y), (m_x_r, m_y_r), (0,255,0), 3)

        pred = model(input.unsqueeze(0))[0].detach()
        #print(pred)
        pred_t = pred[:2]
        top_x, top_y = int(pred_t[0]*w), int(pred_t[1]*h)
        res = cv2.circle(res, (top_x, top_y), 2, (0,0,255), 3)

        pred_m = pred[2:]
        #pred_m = model_m(input.unsqueeze(0))[0].detach()
        #print(pred_m)

        m = pred_t - pred_m * 0.5
        m_x_r, m_y_r = int(m[0]*w), int(m[1]*h)
        res = cv2.line(res, (top_x, top_y), (m_x_r, m_y_r), (0,0,255), 2)

        '''
        pred_l = model_l(input.unsqueeze(0))[0].detach()

        l = pred + pred_l * 0.2
        left_x_r, left_y_r = int(l[0]*w), int(l[1]*h)
        res = cv2.circle(res, (left_x_r, left_y_r), 2, (255,0,0), 3)


        pred_r = model_r(input.unsqueeze(0))[0].detach()

        r = pred + pred_r * 0.2
        right_x_r, right_y_r = int(r[0]*w), int(r[1]*h)
        res = cv2.circle(res, (right_x_r, right_y_r), 2, (255,0,0), 3)
        '''
        img[offset[1]:offset[1]+300,offset[0]:offset[0]+300] = res
        #cv2.imshow("res", img)
        cv2.imwrite("output/{}.jpg".format(str(count).zfill(4)),img)
        #cv2.waitKey(0)
        if count > 200:
            break
