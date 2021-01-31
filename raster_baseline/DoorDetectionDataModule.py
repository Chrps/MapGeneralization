from warnings import warn

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataset import DoorDataset


def _collate_fn(batch):
    return tuple(zip(*batch))


CLASSES = (
    "door",
)

'''
def _prepare_voc_instance(image, target):
    """
    Prepares VOC dataset into appropriate target for fasterrcnn

    https://github.com/pytorch/vision/issues/1097#issuecomment-508917489
    """
    anno = target["annotation"]
    h, w = anno["size"]["height"], anno["size"]["width"]
    boxes = []
    classes = []
    area = []
    iscrowd = []
    objects = anno["object"]
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        bbox = obj["bndbox"]
        bbox = [int(bbox[n]) - 1 for n in ["xmin", "ymin", "xmax", "ymax"]]
        boxes.append(bbox)
        classes.append(CLASSES.index(obj["name"]))
        iscrowd.append(int(obj["difficult"]))
        area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    classes = torch.as_tensor(classes)
    area = torch.as_tensor(area)
    iscrowd = torch.as_tensor(iscrowd)

    image_id = anno["filename"][5:-4]
    image_id = torch.as_tensor([int(image_id)])

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target["image_id"] = image_id

    # for conversion to coco api
    target["area"] = area
    target["iscrowd"] = iscrowd

    return image, target
'''

class DoorDetectionDataModule(LightningDataModule):
    name = "doordetection"

    def __init__(
        self,
        data_dir: str,
        year: str = "2020",
        num_workers: int = 12,
        *args,
        **kwargs,
    ):
        """
        TODO(teddykoker) docstring
        """
        super().__init__(*args, **kwargs)

        self.year = year
        self.data_dir = data_dir
        self.num_workers = num_workers

    @property
    def num_classes(self):
        """
        Return:
            1
        """
        return 1

    def train_dataloader(self, batch_size=1, transforms=None):
        """
        VOCDetection train set uses the `train` subset

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        dataset = DoorDataset(root='/home/markpp/github/MapGeneralization/data/Public',
                              list='train_list_reduced_cnn.txt')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return loader

    def val_dataloader(self, batch_size=1, transforms=None):
        """
        VOCDetection val set uses the `val` subset

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """

        dataset = DoorDataset(root='/home/markpp/github/MapGeneralization/data/Public',
                              list='valid_list_reduced_cnn.txt')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return loader
