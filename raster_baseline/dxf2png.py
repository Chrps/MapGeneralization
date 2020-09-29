import os
import argparse
import numpy as np
import cv2

from crop import crop_using_cnt, crop_using_min_max, map_bbs_to_crop, Mouse

# note that 'simple' and 'advanced' somethimes fails to crop correctly
# 'simple'  uses a faulty rendering of arcs etc., leading to mapping problems
# 'advanced'  uses a rendering which includes text, leading to mapping problems
# 'manual'  takes some trial-and-error but should produce correct mapping

methods = ['simple', 'advanced', 'manual']

def get_bbs_from_file(path):
    boxes_file = open(path,"r")
    bb_lines = boxes_file.readlines()
    bbs = []
    for bb_line in bb_lines:
        x1, y1, x2, y2 = bb_line.split(' ')
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        bbs.append([x1, y1, x2-x1, y2-y1])
    return bbs

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
    ap.add_argument("-r", "--root", type=str,
                    default='/home/markpp/github/MapGeneralization/data/Public/', help="data root dir")
    ap.add_argument("-l", "--list", type=str,
                    default='test_list.txt', help="File list")
    ap.add_argument("-m", "--method", type=int,
                    default=1, help="method selection")
    args = vars(ap.parse_args())

    METHOD = methods[args["method"]]

    file_list = []
    with open(os.path.join(args["root"],args["list"])) as f:
        file_list = f.read().splitlines()

    for path in file_list[:]:

        image_path = os.path.join(args["root"],path.replace('/anno/','/images/').replace('_w_annotations.gpickle','.png'))
        print(os.path.dirname(image_path))
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))

        crop_path = os.path.join(args["root"],path.replace('/anno/','/pdfs/').replace('_w_annotations.gpickle','.txt'))

        if os.path.exists(crop_path):
            continue

        if METHOD == 'manual':
            pdf_path = os.path.join(args["root"],path.replace('/anno/','/pdfs/').replace('_w_annotations.gpickle','.pdf'))
            os.system("gs -sDEVICE=png256 -r300 -dNOPAUSE -dBATCH -dSAFER -sOutputFile=temp.png {}".format(pdf_path))

            bb_path = os.path.join(args["root"],path.replace('/anno/','/bboxes/').replace('_w_annotations.gpickle','_boxes_image_format.txt'))
            bbs = get_bbs_from_file(bb_path)

            img = cv2.imread("temp.png")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #resize to hight of screen
            scale = 1080 / img.shape[1]
            scale_ = img.shape[1] / 1080
            tmp_img = cv2.resize(img_gray, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            ret, threshold = cv2.threshold(tmp_img, 254, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3,3), np.uint8)
            tmp_img = cv2.dilate(threshold, kernel, iterations=1)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)

            # Create a window
            cv2.namedWindow('select crop',cv2.WINDOW_AUTOSIZE)
            mouse = Mouse()
            cv2.setMouseCallback('select crop',mouse.get_click)
            x, y, w, h = 0, 0, 0, 0
            while(1):
                tmp = tmp_img.copy()
                if mouse.left and mouse.middle:
                    tmp = cv2.rectangle(tmp,(mouse.l[0], mouse.l[1]),(mouse.m[0], mouse.m[1]),(0,255,0),1)
                    x, y, w, h = int(mouse.l[0]*scale_), int(mouse.l[1]*scale_), int((mouse.m[0]-mouse.l[0])*scale_), int((mouse.m[1]-mouse.l[1])*scale_)
                    crop = img[y:y+h, x:x+w]
                    crop_gray = img_gray[y:y+h, x:x+w]
                    crop_w_bbs = map_bbs_to_crop(crop_gray, bbs)
                    cv2.imshow("crop_w_bbs",crop_w_bbs)

                cv2.imshow("select crop",tmp)
                key = cv2.waitKey(30)
                if key == 27:
                    break
            # save crop
            crop_file = open(crop_path,"w")
            crop_file.write("{},{},{},{}".format(x, y, w, h))
            crop_file.close()

        if METHOD == 'advanced':
            pdf_path = os.path.join(args["root"],path.replace('/anno/','/pdfs/').replace('_w_annotations.gpickle','.pdf'))
            os.system("gs -sDEVICE=png256 -r300 -dNOPAUSE -dBATCH -dSAFER -sOutputFile=temp.png {}".format(pdf_path))

            bb_path = os.path.join(args["root"],path.replace('/anno/','/bboxes/').replace('_w_annotations.gpickle','_boxes_image_format.txt'))
            bbs = get_bbs_from_file(bb_path)

            img = cv2.imread("temp.png")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x, y, w, h = crop_using_cnt(img_gray)
            crop = img[y:y+h, x:x+w]
            crop_gray = img_gray[y:y+h, x:x+w]
            if(crop_gray.shape[0] < 1):
                continue
            crop_w_bbs = map_bbs_to_crop(crop_gray, bbs)

            # save crop
            crop_file = open(crop_path,"w")
            crop_file.write("{},{},{},{}".format(x, y, w, h))
            crop_file.close()

        if METHOD == 'smart':
            dxf_path = os.path.join(args["root"],path.replace('/anno/','/dxf/').replace('_w_annotations.gpickle','.dxf'))
            print(dxf_path)

            os.system("libreoffice --headless --convert-to pdf {}".format(dxf_path))
            pdf_path = os.path.basename(dxf_path.replace('.dxf','.pdf'))
            print(pdf_path)

            os.system("pdfcrop {}".format(pdf_path))
            pdf_crop_path = pdf_path.replace('.pdf','-crop.pdf')
            print(pdf_crop_path)



            os.system("gs -sDEVICE=png256 -r600 -dNOPAUSE -dBATCH -dSAFER -sOutputFile={} {}".format(image_path,pdf_crop_path))

            '''
            bb_path = os.path.join(args["root"],path.replace('/anno/','/bboxes/').replace('_w_annotations.gpickle','_boxes_image_format.txt'))
            print(bb_path)

            bbs = get_bbs_from_file(bb_path)

            crop, crop_gray, crop_w_bbs = map_bbs_to_crop(bbs, cv2.imread(image_path))
            cv2.imshow("crop",crop)
            cv2.imshow("crop_w_bbs",crop_w_bbs)
            cv2.waitKey(0)

            os.system("rm {}".format(pdf_path))
            os.system("rm {}".format(pdf_crop_path))
            '''
        #'''
        cv2.imwrite(image_path, crop) # already pressent due to gs command
        cv2.imwrite(image_path.replace('.png','_gray.png'), crop_gray)
        cv2.imwrite(image_path.replace('.png','_bbs.png'), crop_w_bbs)

        os.system("rm temp.png")
        #'''
