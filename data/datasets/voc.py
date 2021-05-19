
import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from core.boxops.bndbox import BoxList
from utils.utils import resize_eq_ratio
import numpy as np

class PascalVOCDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, cfg, transforms=None ,image_loader=None):
        self.root = cfg.DATASETS.IMG_DIR
        self.image_set = 'train'
        self.keep_difficult = cfg.DATASETS.USE_DEFFICULT
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return index ,img, target
    def collate_fn(self, batch):
        paths, imgs, targets = zip(*batch)
        new_targets = []
        new_imgs = []
        img_size = self.multiscale[np.random.choice([i for i in range(len(self.multiscale))], p=self.scale_weight)]
        for path ,img ,target in zip(paths ,imgs, targets):
            h_factor, w_factor = img.shape[1] ,img.shape[2]
            # Pad to square resolution

            img, _scale, pad = resize_eq_ratio(img, img_size)

            if self.remove_small:
                target.remove_small()
            boxes = target.bbox
            # Extract coordinates for unpadded + unscaled image
            w_factor *= _scale
            h_factor *= _scale
            # Adjust for added padding
            x1 = w_factor * (boxes[:, 0] - boxes[:, 2] / 2) + pad[0]
            y1 = h_factor * (boxes[:, 1] - boxes[:, 3] / 2) + pad[2]
            x2 = w_factor * (boxes[:, 0] + boxes[:, 2] / 2) + pad[1]
            y2 = h_factor * (boxes[:, 1] + boxes[:, 3] / 2) + pad[3]
            boxes[:, 0] = ((x1 + x2) / 2) / img_size[0]
            boxes[:, 1] = ((y1 + y2) / 2) / img_size[1]
            boxes[:, 2] *= w_factor / img_size[0]
            boxes[:, 3] *= h_factor / img_size[1]
            target.size = (img_size[0] ,img_size[1])
            target.add_field('sample_per_cls', self.class_nums)
            new_targets.append(target)
            new_imgs.append(img)
        new_imgs = torch.stack(new_imgs)
        self.batch_count += 1
        return paths, new_imgs, new_targets

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        if self.cfg.DATASETS.USE_BACKGROUND:
            target.add_field("labels", anno["labels"])
        else:
            target.add_field("labels", anno["labels"] - 1)

        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]
if __name__ == '__main__':
    from config import cfg
    import cv2
    torch.cuda.set_device(7)
    cfg.merge_from_file('config/pelee64_darkfpn_voc.yml')
    # cfg.merge_from_list(opts)
    cfg.freeze()
    COLOR = [(255 ,0 ,0) ,(0 ,255 ,0) ,(0 ,0 ,255) ,(255 ,0 ,255)]
    # datasets = DarknetDataset(cfg, transforms.ToTensor())
    datasets = PascalVOCDataset(cfg, transforms.ToTensor())
    # datasets.cal_class_info()
    val_loader = torch.utils.data.DataLoader(datasets,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=False,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=datasets.collate_fn)
    for path, imgs, targets in val_loader:
        for i, (img, target) in enumerate(zip(imgs, targets)):
            img_size = img.shape
            img_size = img_size[1:]
            cv_img = transforms.ToPILImage()(img)
            cv_img = np.array(cv_img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            labels = target.get_field('labels').long()
            for j ,box in enumerate(target.bbox):
                if box[0] == -1:
                    break
                x = int((box[0] - box[2] / 2) * img_size[1])
                y = int((box[1] - box[3] / 2) * img_size[0])
                x2 = int((box[2] / 2 + box[0]) * img_size[1])
                y2 = int((box[3] / 2 + box[1]) * img_size[0])
                cv2.rectangle(cv_img, (int(x), int(y)), (x2, y2), COLOR[labels[j]], 1)
                # cv2.putText(cv_img, str(round(box[4], 2)), (x + 6, y + 6),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR[int(box[-1])], 1, cv2.LINE_AA)
            cv2.imshow("cv_img_{}:".format(i), cv_img)
            # cv2.imwrite("./cv_img.jpg",cv_img)
        cv2.waitKey(-1)
