import cv2
import torch
import random
import numbers
import numpy as np


class CvColorJitter(object):
    """
    :param brightness, contrast and saturation of an image
    :return images after transforms
    """

    def __init__(self, brightness=0, contrast=0, staturation=0, hue=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.staturation = self._check_input(staturation, "staturation")
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a singer number, it must be non negative.".fromat(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}.".format(bound[0], bound[1]))
        else:
            raise TypeError("{} should be a single0.229, value, or a list/tuple with lenght 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    def _distort(self, image, alpha=1.0, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    def _convert(self, img):

        # convert brightness
        if random.randrange(2):
            self._distort(img, beta=random.uniform(-32, 32))

        # convert contrast
        if random.randrange(2):
            self._distort(img, alpha=random.uniform(self.contrast[0], self.contrast[1]))

        # convert staturation and hue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if random.randrange(2):
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.uniform(-10, 10)) % 180
        if random.randrange(2):
            self._distort(img[:, :, 1], alpha=random.uniform(-0.5, 1.5))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        return img

    def __call__(self, img, target=None):
        """
        :param img:
        :param target:
        :return:
            cv image: Color jittered image.
        """

        return self._convert(img), target


class CvToTensor(object):
    def __call__(self, img, target=None):
        img = torch.from_numpy(img)
        return img, target


class CvNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean[::-1]
        self.std = std[::-1]

    def __call__(self, tensor, target=None):
        tensor = tensor.astype(np.float32) / 255.0
        tensor -= self.mean
        tensor /= self.std
        tensor = tensor.transpose(2, 0, 1)
        return tensor, target


class Compose(object):
    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, image, target=None):
        for trans in self.transforms:
            image, target = trans(image, target)

        return image, target


if __name__ == "__main__":
    image_path = "./example.jpg"
    cv_image = cv2.imread(image_path)

    Normlize = CvNormalize([0.406, 0.485, 0.456], [0.224, 0.225, 0.229])

    transform = Compose([CvColorJitter(0.4, 0.4, 0.2, (-0.5, 0.5))])
    img_trans = transform(cv_image, None)
    print(img_trans[0].shape)
    cv2.imwrite("trans_image_hue0.jpg", img_trans[0])
    # cv2.waitKey(0)