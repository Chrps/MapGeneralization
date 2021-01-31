import torch
import numpy as np
import onnx
import onnxruntime as onnxrt
import numpy as np
import os

import cv2

#from models.common import *

def predict(model, ximg, confidence_threshold=0.1):
    #tmp = np.rollaxis(crop, 2, 0) / 255
    #ximg = tmp[np.newaxis, :, :, :].astype(np.float32)
    #print("The shape of the Input is: ", ximg.shape)
    result = model(ximg)
    #probs = result[0].ravel()
    #boxes = result[1]
    print(result[0][0].shape)
    print(result[1][0].shape)
    print(result[2][0].shape)

    crop_h, crop_w, _ = crop.shape
    confs, bbs = [], []
    dets = np.where(probs > confidence_threshold)[0] # threshold for performance reasons
    for det in dets:
        confidence = probs[det]
        confs.append(float(confidence))
        detection = boxes[det]
        center_x, center_y = int(detection[0] * crop_w), int(detection[1] * crop_h)
        width, height = int(detection[2] * crop_w), int(detection[3] * crop_h)
        left, top = int(center_x - width / 2), int(center_y - height / 2)
        bbs.append([left, top, width, height])
    return confs, bbs

def onnx_predict(ximg, offsets, confidence_threshold=0.1):
    #ximg = ximg[np.newaxis, :, :, :].astype(np.float32)

    #print("The shape of the Input is: ", ximg.shape)
    result = sess.run(None, {input_name: ximg})

    # single
    #probs = result[0].ravel()
    #boxes = result[1]

    confs, bbs = [], []

    _, _, crop_h, crop_w = ximg.shape

    for probs, boxes, offset in zip(result[0],result[1],offsets):
        confs_, bbs_ = [], []
        dets = np.where(probs > confidence_threshold)[0] # threshold for performance reasons
        for det in dets:
            confidence = probs[det]
            confs_.append(float(confidence))
            detection = boxes[det]
            center_x, center_y = int(detection[0] * crop_w), int(detection[1] * crop_h)
            width, height = int(detection[2] * crop_w), int(detection[3] * crop_h)
            left, top = int(center_x - width / 2), int(center_y - height / 2)
            bbs_.append([left+offset[0], top+offset[1], width, height])

        # non-max supression #TODO: more advanced version (merge)? the
        nms_indices = cv2.dnn.NMSBoxes(bbs_, confs_, confThreshold, nmsThreshold)

        for indice in nms_indices:
            confs.append(confs_[indice[0]])
            bbs.append(bbs_[indice[0]])
    return confs, bbs


class pydetect:
    def __init__(self,model_path,weight_path,show=False):
        self.show = show

        self.bboxes = []
        self.confidences = []
        # Initialize the parameters
        self.confThreshold = 0.01  #Confidence threshold
        self.nmsThreshold = 0.2   #Non-maximum suppression threshold
        self.inpWidth = 416       #Width of network's input image
        self.inpHeight = 416      #Height of network's input image

        self.net = cv2.dnn.readNetFromDarknet(model_path, weight_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self,img):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.inpWidth, self.inpHeight), swapRB=True, crop=False)

        print(blob.shape)
        # Sets the input to the network
        self.net.setInput(blob)

        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected outputs
        outputsNames = [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        print(outputsNames)
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(outputsNames)

        # Remove the bounding boxes with low confidence
        self.postprocess(img, outs)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        self.bboxes = []
        self.confidences = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            self.bboxes.append([left, top, left + width, top + height])
            self.confidences.append(confidences[i])


if __name__ == "__main__":

    crop_size = 640#416
    confThreshold = 0.1 # raise to remove false positives
    nmsThreshold = 0.25 # boxes are determined redundant if IOU value greater than 0.4
    #batch_size = 32


    #weights = "yolov3-tiny-bo.pt"
    #model = torch.load(weights)['model']
    #model.eval()


    #sess = onnxrt.InferenceSession("yolov3-tiny-bo.onnx") #yolov3-tiny-bo.onnx
    #print("The model expects input shape: ", sess.get_inputs()[0].shape)
    #input_name = sess.get_inputs()[0].name
    #label_name = sess.get_outputs()[0].name


    modelConfiguration = "yolov3-tiny-bo.cfg"
    modelWeights = "yolov3-tiny-bo_best.weights"
    #modelConfiguration = "/home/dmri/github/dockerYolo/code/darknet/cfg/yolov3-tiny-1class.cfg"
    #modelWeights = "/home/dmri/github/dockerYolo/code/darknet/trainedWeights/tinyv3/yolov3-tiny-1class_12000.weights"

    det = pydetect(modelConfiguration, modelWeights, show = False)

    root_dir = '/home/markpp/github/MapGeneralization/data/Public'

    list = 'valid_list_reduced_cnn.txt'

    with open(os.path.join(root_dir,list)) as f:
        lines = f.read().splitlines()

    img_list = []
    for path in lines:
        img_list.append(os.path.join(root_dir,path))


    for img_path in img_list[:1]:
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(), cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # (1) pad the image such that it is divisable by crop_size
        right_pad = img_w % crop_size

        bottom_pad = img_h % crop_size

        img = cv2.copyMakeBorder(img, top=0, bottom=bottom_pad, left=0, right=right_pad, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

        offsets,batch = [],[]
        for x in range(0,img.shape[1]-crop_size,crop_size//2):
            for y in range(0,img.shape[0]-crop_size,crop_size):
                print("x {}, y {}".format(x,y))
                crop = img[y:y+crop_size, x:x+crop_size].copy()
                #crop = cv2.resize(crop, (416,416))

                #tmp = np.rollaxis(crop, 2, 0) / 255.0
                #tmp = np.expand_dims(tmp, axis=0)
                #batch.append(tmp)
                #offsets.append([x,y])

                #confs, bbs = onnx_predict(np.array(tmp, dtype=np.float32),offsets)

                det.detect(crop)

                print(det.confidences)

                cv2.imshow("crop",crop)
                cv2.waitKey(100)

        cv2.imshow("img",img)
        cv2.waitKey()


'''
    # (2) determine how manyn times the image should be cropped
    N = math.ceil((img_h * img_w) / (dataset.image_size * dataset.image_size)) * 2

    for n in range(N):
        if img_w - (dataset.image_size-1) > 0 and img_h - (dataset.image_size-1) > 0:
            # (3) select a random x_min and y_min values inside the image
            x_min = random.randrange(0, img_w - (dataset.image_size-1))
            y_min = random.randrange(0, img_h - (dataset.image_size-1))
            x_max, y_max = x_min+dataset.image_size, y_min+dataset.image_size
            crop = img[y_min:y_max, x_min:x_max].copy()


            bbs = []
            # (4) check if bounding boxes are indside the crop
            for bb in boxes:
                #if bb intersects the crop rect
                if bb[0] > x_min and bb[1] > y_min and bb[2] < x_max and bb[3] < y_max:
                    bb_x_min, bb_y_min = bb[0]-x_min, bb[1]-y_min
                    bb_x_max, bb_y_max = bb[2]-x_min, bb[3]-y_min
                    #crop = cv2.rectangle(crop,(bb_x_min, bb_y_min),(bb_x_max, bb_y_max),(0,0,255),2)
                    bbs.append([bb_x_min/dataset.image_size,
                                bb_y_min/dataset.image_size,
                                bb_x_max/dataset.image_size,
                                bb_y_max/dataset.image_size])

    batch = []

    # iterate over each item
    for item in items[:]:
        frames = []

        img_paths = sorted([f for f in os.listdir(os.path.join(root_dir,item,"rgb")) if f.endswith('.jpg')])
        print("{} has {} images".format(item,len(img_paths)))
        # iterate over each image belonging to the given item
        for i,img_file in enumerate(img_paths[:]):
            pred_bb, pred_cls, pred_conf = [], [], []
            gt_bb, gt_cls = [], []

            img = cv2.imread(os.path.join(root_dir,item,"rgb",img_file))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print("image {} not loaded".format(img_file))
                continue

            mask_file = img_file.replace('jpg','png')
            mask = cv2.imread(os.path.join(root_dir,item, "masks",mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
                print("empty mask created in place of missing file {}".format(mask_file))


            new_mask  = label_unique(mask)
            #cv2.imwrite(os.path.join(root_dir,item,"results",mask_file),new_mask)

            gts = mask2gt(new_mask)
            for gt in gts:
                gt_bb.append(gt) #xmin, ymin, xmax, ymax
                gt_cls.append(0)
            # divide image into appropritate crops and detect defects
            for x in range(0,img.shape[1]-crop_size,crop_size):
                for y in range(0,img.shape[0]-crop_size,crop_size):
                    crop = img[y:y+crop_size, x:x+crop_size]
                    batch.append(np.rollaxis(crop, 2, 0) / 255)

                    if len(batch) == batch_size:
                        confs, bbs = predict(model, np.array(batch, dtype=np.float32))

                        for conf, bb in zip(confs, bbs):
                            left, top, width, height = x + bb[0], y + bb[1], bb[2], bb[3] # add crop offset
                            img = cv2.rectangle(img, (left, top), (left + width, top + height), (0,255,0), 2)
                            cv2.putText(img, "{:.2f}".format(conf), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                            pred_bb.append([float(left)/img.shape[1], float(top)/img.shape[0],
                                            float(left+width)/img.shape[1], float(top+height)/img.shape[0]])
                            pred_cls.append(0)
                            pred_conf.append(conf)
                        batch = []

            cv2.imwrite(os.path.join(root_dir,item,"results",img_file),img)
            frames.append((np.array(pred_bb), np.array(pred_cls), np.array(pred_conf),
                           np.array(gt_bb), np.array(gt_cls)))

            pred_name = img_file.replace('jpg','txt')
            pred_path = os.path.join(root_dir,"preds",pred_name) #os.path.join(root_dir,item,"pred",pred_name)
            if os.path.isfile(pred_path):
                with open(pred_path, 'w') as pred_file:
                    for conf, bb in zip(pred_conf,pred_bb):
                        pred_file.write('defect {:.3f} {} {} {} {}\n'.format(conf,
                                                                         int(bb[0]*img.shape[1]),
                                                                         int(bb[1]*img.shape[0]),
                                                                         int(bb[2]*img.shape[1]),
                                                                         int(bb[3]*img.shape[0])))
            else:
                np.savetxt(pred_path, [], delimiter=",", fmt='%u')

            gt_name = img_file.replace('jpg','txt')
            gt_path = os.path.join(root_dir,"gts",gt_name) #os.path.join(root_dir,item,"gt",gt_name)
            if os.path.isfile(gt_path):
                with open(gt_path, 'w') as gt_file:
                    for bb in gt_bb:
                        gt_file.write('defect {} {} {} {}\n'.format(int(bb[0]*img.shape[1]),
                                                                    int(bb[1]*img.shape[0]),
                                                                    int(bb[2]*img.shape[1]),
                                                                    int(bb[3]*img.shape[0])))
            else:
                np.savetxt(gt_path, [], delimiter=",", fmt='%u')


    import matplotlib.pyplot as plt

    frames = np.load("frames.npy", allow_pickle=True)
    print(frames[0])

    mAP = DetectionMAP(n_class=1)
    for i, frame in enumerate(frames):
        print("Evaluate frame {}".format(i))
        #show_frame(*frame)
        mAP.evaluate(*frame)

    mAP.plot()
    plt.show()
    '''
