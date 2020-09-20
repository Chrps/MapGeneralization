import os
import argparse
import numpy as np
import cv2

scale = 1.0

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
    Scale images and corresponding annotations. NB overwrites the origianls!
    Command:
        -l path/to/image_list.txt
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--list", type=str,
                    help="File list")
    args = vars(ap.parse_args())

    file_list = []
    with open(args["list"]) as f:
        file_list = f.read().splitlines()

    for img_path in file_list[2:]:
                bb_path = path.replace('_w_annotations.gpickle','')

        bbs = get_bbs_from_file("A1322PE-0_boxes_image_format.txt")

        # read, scale and overwrite color image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        points = np.argwhere(np.less(img_gray, 255))
        points = np.fliplr(points)
        x, y, w, h = cv2.boundingRect(points)
        img_gray = img_gray[y:y+h, x:x+w]
        img = img[y:y+h, x:x+w]

        bounding_boxes = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for bb in bbs:
            h_pixels, w_pixels, _ = bounding_boxes.shape
            x1, y1, x2, y2 = int(bb[0]*w_pixels), int(bb[1]*h_pixels), int((bb[0]+bb[2])*w_pixels), int((bb[1]+bb[3])*h_pixels)
            bounding_boxes = cv2.rectangle(bounding_boxes,(x1, y1),(x2, y2),(0,255,0),1)

        cv2.imshow("crop", img)
        cv2.imwrite(img_path,img)

        binary = img_gray[np.where(np.less(img_gray, 255))] = 0
        cv2.imshow("binary", binary)
        cv2.imwrite(img_path,img)

        cv2.imshow("bounding_boxes", bounding_boxes)
        cv2.imwrite(img_path,img)

        cv2.waitKey(0)
        '''

        # read, scale and overwrite depth image
        depth_path = img_path[:-7] + "depth.jpg"
        depth = cv2.resize(cv2.imread(depth_path), (int(IMG_W*scale), int(IMG_H*scale)))
        cv2.imwrite(depth_path,depth)

        # read org annotations
        poi_path = img_path[:-7] + "ear_pos.csv"
        pois = []
        with open(poi_path) as poi_file:
            points = []
            for poi in poi_file.read().splitlines():
                x, y = poi.split(",")
                points.append([int(x),int(y)])
            pois.append(points)

        # overwrite with scaled annotations
        with open(poi_path, 'w') as out_file:
            for poi in pois:
                for p in poi:
                    out_file.write("{},{}\n".format(int(p[0]*scale),int(p[1]*scale)))
        '''
