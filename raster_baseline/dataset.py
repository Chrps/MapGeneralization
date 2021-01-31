import os
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms.transforms import RandomCrop, ToTensor, Normalize, Compose
import random
import math

import cv2

class DoorDataset(torch.utils.data.Dataset):
    def __init__(self, root, list):
        self.root = root
        self.image_size = 640
        self.img_list = []
        self.crops = []
        self.boxes = []

        with open(os.path.join(root,list)) as f:
            self.list = f.read().splitlines()

        for path in self.list:
            self.img_list.append(os.path.join(root,path))


    def load_sample(self, idx):
        img = cv2.imread(self.img_list[idx])
        #img = cv2.cvtColor(), cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        boxes_path = self.img_list[idx].replace('/images/','/bboxes/').replace('.png','_boxes_image_format.txt')
        with open(boxes_path,"r") as boxes_file:
            bb_lines = boxes_file.readlines()

        boxes = []
        for bb_line in bb_lines:
            x2, y2, x1, y1 = bb_line.split(' ')
            x1, y1, x2, y2 = int(float(x1)*img_w), int(float(y1)*img_h), int(float(x2)*img_w), int(float(y2)*img_h)
            boxes.append([x1, y1, x2, y2])

        return img, boxes, None


    '''
    def load_sample(self, idx):
        while len(self.crops) < 1:
            #print(self.img_list[idx])
            img = cv2.cvtColor(cv2.imread(self.img_list[idx]), cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]


            boxes_path = self.img_list[idx].replace('/images/','/bboxes/').replace('.png','_boxes_image_format.txt')
            with open(boxes_path,"r") as boxes_file:
                bb_lines = boxes_file.readlines()

            boxes = []
            for bb_line in bb_lines:
                x2, y2, x1, y1 = bb_line.split(' ')
                x1, y1, x2, y2 = int(float(x1)*img_w), int(float(y1)*img_h), int(float(x2)*img_w), int(float(y2)*img_h)
                boxes.append([x1, y1, x2, y2])
            # (1) check that there is room to crop the image
            if self.image_size >= img_w:
                right_pad = self.image_size - img_w
            else:
                right_pad = 0
            if self.image_size >= img_h:
                bottom_pad = self.image_size - img_h
            else:
                bottom_pad = 0
            img = cv2.copyMakeBorder(img, top=0, bottom=bottom_pad, left=0, right=right_pad, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

            # (2) determine how manyn times the image should be cropped
            N = math.ceil((img_h * img_w) / (self.image_size * self.image_size))

            for n in range(N):
                # (3) select a random x_min and y_min values inside the image
                #print(img_h - self.image_size)
                x_min = random.randint(1, img_w - self.image_size) - 1
                y_min = random.randint(1, img_h - self.image_size) - 1
                x_max, y_max = x_min+self.image_size, y_min+self.image_size
                crop = img[y_min:y_max, x_min:x_max].copy()

                bbs, labs = [], []
                # (4) check if bounding boxes are indside the crop
                for bb in boxes:

                    #if bb intersects the crop rect
                    if bb[0] > x_min and bb[1] > y_min and bb[2] < x_max and bb[3] < y_max:
                        bb_x_min, bb_y_min = bb[0]-x_min, bb[1]-y_min
                        bb_x_max, bb_y_max = bb[2]-x_min, bb[3]-y_min
                        bbs.append([bb_x_min, bb_y_min, bb_x_max, bb_y_max])
                        #self.labs.append('door')

                if len(bbs) > 0:
                    self.crops.append(crop)
                    self.boxes.append(bbs)
                    #self.labels.append(labs)

        bboxes = self.boxes.pop()
        classes = []
        area = []
        iscrowd = []
        for bbox in bboxes:
            classes.append(0)
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = classes

        # for conversion to coco api
        target["area"] = area
        return self.crops.pop(), target

        '''
    def __getitem__(self, idx):
        img, boxes, labels = self.load_sample(idx)

        '''
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img[0] = (img[0] - 0.485)/0.229
        img[1] = (img[1] - 0.456)/0.224
        img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img)
        '''
        return img, boxes

    def __len__(self):
        return len(self.img_list)



if __name__ == '__main__':

    dataset = DoorDataset(root='/home/markpp/github/MapGeneralization/data/Public',
                          list='valid_list_reduced_cnn.txt')

    output_dir = "yolo/val"

    for idx in range(len(dataset)):
        img, boxes = dataset[idx]
        img_h, img_w = img.shape[:2]

        # (1) check that there is room to crop the image
        if dataset.image_size >= img_w:
            right_pad = dataset.image_size - img_w
        else:
            right_pad = 0
        if dataset.image_size >= img_h:
            bottom_pad = dataset.image_size - img_h
        else:
            bottom_pad = 0
        img = cv2.copyMakeBorder(img, top=0, bottom=bottom_pad, left=0, right=right_pad, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

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

                if 1:#len(bbs) > 0:
                    filename = "idx-{}_n-{}_x_min-{}_y_min-{}.jpg".format(idx, n, x_min, y_min)
                    cv2.imwrite(os.path.join(output_dir,filename),crop)
                    with open(os.path.join(output_dir,filename.replace('jpg','txt')), 'w') as pred_file:
                        for bb in bbs:
                            pred_file.write('0 {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(bb[0],bb[1],bb[2],bb[3]))


'''
    data = iter(dataloader)
    labels = []
    for batch in range(10):
        images,targets = next(data)
        for i, img_tar in enumerate(zip(images,targets)):
            img, tar = img_tar
            labels.append(tar.item())
            img = unnormalize(img)
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            if batch == 0:
                cv2.imwrite("output/b{}_i{}_c{}.png".format(batch,i,tar),img)
        print("# defects {}, ok {}".format(np.count_nonzero(np.array(labels) == 0),np.count_nonzero(np.array(labels) == 1)))

class BOItemDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.image_size = 240
        self.transforms = self.get_transform()
        self.classes = ['defect','ok']

        self.img_list = []
        self.list_idx = 0

        items = sorted([f for f in os.listdir(root) if 'item' in f])
        for item in items[:]:
            for f in [f for f in os.listdir(os.path.join(self.root,item,"rgb")) if f.lower().endswith(('.jpg', '.jpeg'))]:
                self.img_list.append(os.path.join(self.root,item,"rgb",f))

        # count number of samples
        width, height, _ = cv2.imread(os.path.join(self.root, self.img_list[0])).shape
        self.n_samples = len(self.img_list) * height // self.image_size * width // self.image_size
        print("number of patches {}".format(self.n_samples))

        self.crops = []
        self.labels = []

    def load_sample(self, idx):
        # generate crops from image, label as defect if it contains a defect
        if len(self.crops) < 1:
            rgb_path = os.path.join(self.root, self.img_list[self.list_idx])
            self.list_idx = self.list_idx + 1
            mask_path = rgb_path.replace('rgb','masks').replace('jpg','png')


            # use PIL to load data
            img = Image.open(rgb_path).convert("RGB")
            mask = Image.open(mask_path)
            if mask is None:
                mask = Image.new("L", img.size, (0,))

            img = cv2.imread(rgb_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.int8)

            # divide image into appropritate crops and detect defects
            for x in range(0,img.shape[1]-self.image_size,self.image_size):
                for y in range(0,img.shape[0]-self.image_size,self.image_size):
                    crop = img[y:y+self.image_size, x:x+self.image_size]
                    mask_crop = mask[y:y+self.image_size, x:x+self.image_size]
                    nonzero = np.count_nonzero(mask_crop)
                    if nonzero:
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
                    self.crops.append(crop)

        return self.crops.pop(), self.labels.pop()

    def get_transform(self):
        tfms = []
        #tfms.append(RandomCrop(self.image_size))
        tfms.append(ToTensor())
        tfms.append(Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        return Compose(tfms)

    def __getitem__(self, idx):
        img, target = self.load_sample(idx)

        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img[0] = (img[0] - 0.485)/0.229
        img[1] = (img[1] - 0.456)/0.224
        img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img)
        img = img.float()

        return img, target

    def __len__(self):
        return self.n_samples
'''
