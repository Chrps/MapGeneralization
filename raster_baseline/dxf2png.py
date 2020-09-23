import os
import argparse
import numpy as np
import cv2

def get_bbs_from_file(path):
    boxes_file = open(path,"r")
    bb_lines = boxes_file.readlines()
    bbs = []
    for bb_line in bb_lines:
        x1, y1, x2, y2 = bb_line.split(' ')
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        bbs.append([x1, y1, x2-x1, y2-y1])
    return bbs

def map_bbs_to_crop(bbs, img, scale = 1.0):

    # read, scale and overwrite color image
    #img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = np.argwhere(np.less(img_gray, 255))
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)
    img_gray = img_gray[y:y+h, x:x+w]
    img = img[y:y+h, x:x+w]

    img_w_bbs = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for bb in bbs:
        h_pixels, w_pixels, _ = img_w_bbs.shape
        x1, y1, x2, y2 = int(bb[0]*w_pixels), int(bb[1]*h_pixels), int((bb[0]+bb[2])*w_pixels), int((bb[1]+bb[3])*h_pixels)
        img_w_bbs = cv2.rectangle(img_w_bbs,(x1, y1),(x2, y2),(0,255,0),1)

    return img, img_gray, img_w_bbs

if __name__ == "__main__":
    """
    #libreoffice --headless --convert-to pdf A1322PE-0.dxf
    #pdfcrop A1322PE-0.pdf
    #gs -sDEVICE=png16m -dNOPAUSE -dBATCH -dSAFER -sOutputFile=A1322PE-0.png A1322PE-0-crop.pdf

    Command:
        -l path/to/image_list.txt
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", type=str,
                    default='/home/markpp/github/MapGeneralization/data/Public/', help="data root dir")
    ap.add_argument("-l", "--list", type=str,
                    default='all.txt', help="File list")
    args = vars(ap.parse_args())

    file_list = []
    with open(os.path.join(args["root"],args["list"])) as f:
        file_list = f.read().splitlines()

    for path in file_list[:5]:
        dxf_path = os.path.join(args["root"],path.replace('/anno/','/dxf/').replace('_w_annotations.gpickle','.dxf'))
        print(dxf_path)

        os.system("libreoffice --headless --convert-to pdf {}".format(dxf_path))
        pdf_path = os.path.basename(dxf_path.replace('.dxf','.pdf'))
        print(pdf_path)

        os.system("pdfcrop {}".format(pdf_path))
        pdf_crop_path = pdf_path.replace('.pdf','-crop.pdf')
        print(pdf_crop_path)

        image_path = dxf_path.replace('/dxf/','/images/').replace('.dxf','.png')
        print(os.path.dirname(image_path))
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))

        os.system("gs -sDEVICE=png256 -r600 -dNOPAUSE -dBATCH -dSAFER -sOutputFile={} {}".format(image_path,pdf_crop_path))

        bb_path = os.path.join(args["root"],path.replace('/dxf/','/bboxes/').replace('_w_annotations.gpickle','_boxes_image_format.txt'))
        print(bb_path)

        bbs = get_bbs_from_file(bb_path)
        #for bb in bbs:

        crop, crop_gray, crop_w_bbs = map_bbs_to_crop(bbs, cv2.imread(image_path))

        cv2.imwrite(image_path.replace('.png','_gray.png'), crop_gray)
        cv2.imwrite(image_path.replace('.png','_bbs.png'), crop_w_bbs)

        os.system("rm {}".format(pdf_path))
        os.system("rm {}".format(pdf_crop_path))
