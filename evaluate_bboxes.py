import os
import argparse
import numpy as np
import cv2

# the method which is to be compared to ground truth
# 'r-cnn'
# 'gnn'

methods = ['gnn','r-cnn']

def get_gts_from_file(path):
    boxes_file = open(path,"r")
    bb_lines = boxes_file.readlines()
    bbs = []
    for bb_line in bb_lines:
        x2, y2, x1, y1 = bb_line.split(' ')
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        #w, h = x2 - x1, y2 - y1
        #if w > 0.0 and h > 0.0:
        bbs.append([x1, y1, x2, y2])
    return bbs

def get_preds_from_file(path, confidence=0.5):
    boxes_file = open(path,"r")
    bb_lines = boxes_file.readlines()
    bbs = []
    for bb_line in bb_lines:
        #x2, y2, x1, y1 = bb_line.split(' ')
        score, x2, y2, x1, y1 = bb_line.split(' ')
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        #w, h = x2 - x1, y2 - y1
        if float(score) > confidence:
            bbs.append([x1, y1, x2, y2])
    return bbs

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim), 1/r

def map_bbs_to_crop(crop, bbs, color=(0,255,0), thickness=1):
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    h_pixels, w_pixels, _ = crop.shape
    for bb in bbs:
        x1, y1, x2, y2 = int(bb[0]*w_pixels), int(bb[1]*h_pixels), int((bb[2])*w_pixels), int((bb[3])*h_pixels)
        crop = cv2.rectangle(crop,(x1, y1),(x2, y2),color,thickness)
    return crop

def get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict WRONG!
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert boxA[0] < boxA[2]
    assert boxA[1] < boxA[3]
    assert boxB[0] < boxB[2]
    assert boxB[1] < boxB[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    bb2_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def compute_fp_tp_old(gts, preds, iou_threshold=0.5):
    ious = []
    tp, fn = 0, 0
    for gt in gts:
        gt_x1, gt_y1, gt_x2, gt_y2 = gt

        match = False
        for pred in preds:
            pred_x1, pred_y1, pred_x2, pred_y2 = pred

            # Find intsec, i.e. valid crop region
            iou = get_iou([pred_x1, pred_y1, pred_x2, pred_y2],
                          [gt_x1, gt_y1, gt_x2, gt_y2])

            if iou == 0.0:
                pass
            elif iou < iou_threshold:
                pass #fn = fn + 1
            else:
                match = True
                ious.append(iou)

        if match:
            tp = tp + 1
        else:
            fn = fn + 1

    fps, tps, fns = len(preds)-(tp + fn), tp, fn
    return fps, tps, fns, iou_threshold

def compute_fp_tp(gts, preds, iou_threshold=0.5):
    ious = []
    fps, tps_det, tps_gt, fns = [], [], [], []

    if len(gts):
        for gt in gts:
            match = False
            x1, y1, x2, y2 = gt
            for pred in list(preds): # iterate over copy of preds
                x1_, y1_, x2_, y2_ = pred
                # Find intsec, i.e. valid crop region
                iou = get_iou([x1_, y1_, x2_, y2_],
                              [x1, y1, x2, y2])
                if iou == 0.0:
                    pass
                elif iou < iou_threshold:
                    pass
                else: # match
                    tps_det.append([pred, iou])
                    match = True
                    preds.remove(pred)
                    break
            if match:
                tps_gt.append([gt, iou])
            else:
                fns.append([gt, 0.0])
        for pred in preds:
            fps.append([pred, 0.0])

    return len(fps), len(tps_det), len(fns), iou_threshold


if __name__ == "__main__":
    """
    # load floor plan crop, ground truth, R-CNN boxes or GNN boxes

    Command:
        -l path/to/image_list.txt
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", type=str,
                    default='data/Public/', help="data root dir")
    ap.add_argument("-l", "--list", type=str,
                    default='test_list_reduced.txt', help="File list")
    ap.add_argument("-m", "--method", type=int,
                    default=0, help="method selection")
    ap.add_argument("-i", "--iou", type=float,
                    default=0.5, help="IoU threshold")
    ap.add_argument("-c", "--conf_trsh", type=float,
                    default=0.95, help="Confidence threshold")
    ap.add_argument("-s", "--show", type=int,
                    default=1, help="show detections")
    args = vars(ap.parse_args())

    METHOD = methods[args["method"]]

    file_list = []
    with open(os.path.join(args["root"],args["list"])) as f:
        file_list = f.read().splitlines()

    total_fps, total_tps, total_fns = 0, 0, 0
    for path in file_list[:]:
        if args["show"]:
            crop_path = os.path.join(args["root"],path.replace('/anno/','/images/').replace('_w_annotations.gpickle','_gray.png'))
            crop = cv2.imread(crop_path, -1)

            #resize to hight of screen
            if crop.shape[0] > crop.shape[1]:
                crop, _ = resize_with_aspect_ratio(crop, height=1920)
            else:
                crop, _ = resize_with_aspect_ratio(crop, width=1920)

        bb_path = os.path.join(args["root"],path.replace('/anno/','/bboxes/').replace('_w_annotations.gpickle','_boxes_image_format.txt'))
        bbs = get_gts_from_file(bb_path)
        print(len(bbs))

        if args["show"]:
            result = map_bbs_to_crop(crop, bbs, thickness=6)

        if METHOD == 'gnn':
            pred_path = os.path.join(args["root"],path.replace('/anno/','/pred_bboxes/').replace('_w_annotations.gpickle','_gnn_boxes_image_format.txt'))
            pred_bbs = get_gts_from_file(pred_path)
            #pred_bbs = get_preds_from_file(pred_path, confidence=args["conf_trsh"])
            print(len(pred_bbs))
            if args["show"]:
                result = map_bbs_to_crop(result, pred_bbs, color=(255,0,0), thickness=2)

        if METHOD == 'r-cnn':
            pred_path = os.path.join(args["root"],path.replace('/anno/','/rcnn_bboxes/').replace('_w_annotations.gpickle','.txt'))
            pred_bbs = get_preds_from_file(pred_path, confidence=args["conf_trsh"])

            if args["show"]:
                result = map_bbs_to_crop(result, pred_bbs, color=(255,0,0), thickness=2)

        #print(bbs)
        fps, tps, fns, iou_threshold = compute_fp_tp(bbs, pred_bbs, iou_threshold=args["iou"])

        #print('{} fps,  {} tps, {} fns, iou threshold: {:.2f}\n'.format(fps, tps, fns, iou_threshold))
        total_fps = total_fps + fps
        total_tps = total_tps + tps
        total_fns = total_fns + fns

        if args["show"]:
            cv2.imshow("result",result)
            key = cv2.waitKey()
            cv2.imwrite("result.jpg",result)

            if key == 27:
                break

    precision = total_tps/(total_tps+total_fps)

    print("total_tps {}, total_fps {}, total_fns {}".format(total_tps, total_fps, total_fns))
    precision = total_tps/(total_tps+total_fps)
    print("precision {:.4f}".format(precision))
    recall = total_tps/(total_tps+total_fns)
    print("recall {:.4f}".format(recall))
    accuracy = (total_tps)/(total_tps+total_fps+total_fns)
    print("accuracy {:.4f}".format(accuracy))
    if precision < 0.01:
        precision += 0.01
    f1 = 2*((precision*recall)/(precision+recall))
    print("f1 {:.4f}".format(f1))
