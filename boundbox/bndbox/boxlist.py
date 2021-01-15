import torch


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="cxcywh"):
        # device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("cxcywh", "xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def convert(self, mode):
        if mode not in ("cxcywh", "xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh' or 'cxcywh")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()

        if mode == 'cxcywh':
            w = xmax - xmin
            h = ymax - ymin
            xmin = xmin + w / 2
            ymin = ymin + h / 2

            bbox = torch.cat((xmin.unsqueeze(-1), ymin.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        elif mode == "xyxy":
            xmin = xmin.unsqueeze(-1)
            ymin = ymin.unsqueeze(-1)
            xmax = xmax.unsqueeze(-1)
            ymax = ymax.unsqueeze(-1)
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 0
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def convert_absolute(self, mode):
        if mode not in ("cxcywh", "xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh' or 'cxcywh")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()

        if mode == 'cxcywh':
            w = xmax - xmin
            h = ymax - ymin
            xmin = xmin + w / 2
            ymin = ymin + h / 2

            bbox = torch.cat((xmin.unsqueeze(-1), ymin.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        elif mode == "xyxy":
            xmin = xmin.unsqueeze(-1)
            ymin = ymin.unsqueeze(-1)
            xmax = xmax.unsqueeze(-1)
            ymax = ymax.unsqueeze(-1)
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 0
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox.bbox[:, 0:3:2] = bbox.bbox[:, 0:3:2] * self.size[0]
        bbox.bbox[:, 1:4:2] = bbox.bbox[:, 1:4:2] * self.size[1]
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "cxcywh":
            xmin = self.bbox[:, 0]
            ymin = self.bbox[:, 1]
            w = self.bbox[:, 2]
            h = self.bbox[:, 3]
            xmin = xmin - w / 2
            ymin = ymin - h / 2
            xmax = xmin + w
            ymax = ymin + h
            return xmin, ymin, xmax, ymax
        elif self.mode == "xyxy":
            # labels = self.bbox[:,0]
            xmin = self.bbox[:, 0]
            ymin = self.bbox[:, 1]
            xmax = self.bbox[:, 2]
            ymax = self.bbox[:, 3]
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 0
            # labels = self.bbox[:,0]
            xmin = self.bbox[:, 0]
            ymin = self.bbox[:, 1]
            w = self.bbox[:, 2]
            h = self.bbox[:, 3]
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def area(self):
        box = self.bbox
        if self.mode == "cxcywh":
            area = box[:, 2] * box[:, 3]
        elif self.mode == "xyxy":
            TO_REMOVE = 0
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def clip_coords(self, remove_empty=True):
        if self.mode != 'xyxy':
            raise ValueError("omly xyxy support clip_to_image")

        TO_REMOVE = 0
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def remove_small(self, rm_size):
        index = torch.gt(self.bbox[:, 2] * self.size[0], rm_size)
        self.bbox = self.bbox[index, :]
        labels = self.get_field('labels')
        assert labels[index].numel() == self.bbox[:, 0].numel()
        self.add_field('labels', labels[index])

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]