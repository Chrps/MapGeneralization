import os
import argparse
import numpy as np
import cv2


class Mouse:
    def __init__(self):
        self.l, self.m, self.r = [0,0], [0,0], [0,0]
        self.right, self.middle, self.left = False, False, False

    # mouse callback function
    def get_click(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.l[0],self.l[1] = x,y
            self.left = True
        if event == cv2.EVENT_MBUTTONDOWN:
            self.m[0],self.m[1] = x,y
            self.middle = True
        if event == cv2.EVENT_RBUTTONDOWN:
            self.r[0],self.r[1] = x,y
            self.right = True

def get_bbs_from_file(path):
    boxes_file = open(path,"r")
    bb_lines = boxes_file.readlines()
    bbs = []
    for bb_line in bb_lines:
        x1, y1, x2, y2 = bb_line.split(' ')
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        bbs.append([x1, y1, x2-x1, y2-y1])
    return bbs

def crop_using_cnt(img_gray):
    cnts = cv2.findContours(img_gray,
                            cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))

    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [cnts[-3]], -3, color=255, thickness=-1)
    img_gray[mask==0] = 255
    points = np.argwhere(np.less(img_gray, 255))
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)
    return x, y, w, h

def crop_using_min_max(img_gray):
    points = np.argwhere(np.less(img_gray, 255))
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)

    return x, y, w, h

def map_bbs_to_crop(crop, bbs):

    img_w_bbs = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for bb in bbs:
        h_pixels, w_pixels, _ = img_w_bbs.shape
        x1, y1, x2, y2 = int(bb[0]*w_pixels), int(bb[1]*h_pixels), int((bb[0]+bb[2])*w_pixels), int((bb[1]+bb[3])*h_pixels)
        img_w_bbs = cv2.rectangle(img_w_bbs,(x1, y1),(x2, y2),(0,255,0),2)

    return img_w_bbs

if __name__ == "__main__":
    """
    #libreoffice --headless --convert-to pdf A1322PE-0.dxf
    #pdfcrop A1322PE-0.pdf
    #gs -sDEVICE=png256 -r600 -dNOPAUSE -dBATCH -dSAFER -sOutputFile=A1322PE-0.png A1322PE-0-crop.pdf

    Command:
        -l path/to/image_list.txt
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
                    default='A1322PE-0.png', help="data root dir")

    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    # read, scale and overwrite color image
    scale = 1.0
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    bbs = get_bbs_from_file(args["image"].replace('.png','_boxes_image_format.txt'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = crop_using_cnt(img_gray)
    crop = img[y:y+h, x:x+w]
    crop_gray = img_gray[y:y+h, x:x+w]
    crop_w_bbs = map_bbs_to_crop(crop_gray, bbs, x, y, w, h)

    #cv2.imshow("crop_gray",crop_gray)
    #cv2.imshow("crop_w_bbs",crop_w_bbs)
    cv2.imwrite('crop_gray.png', crop_gray)
    cv2.imwrite('crop_bbs.png', crop_w_bbs)

    cv2.waitKey(0)
