import torch
import numpy as np
import cv2
import os


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        self.random_value_len = len(transforms)
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
            for k, v in data.items():
                if k in objects and k in data:
                    if transform.__class__ in [AddMask, AddIrregularMask, AddBlindIrregularMask,
                                               AddBlindCombinationIrregularMask, AddBlindColorIrregularMask,
                                               AddBlindRandomIrregularMask, RandomFlip]:
                        data[k] = transform(v, rnd_value)
                    else:
                        data[k] = transform(v)

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomFlip(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img, rnd_value):
        if rnd_value[0] > 0.5:
            img = np.fliplr(img)

        return img


class Resize(object):
    def __init__(self, parameters):
        self.size = parameters['size']

    def __call__(self, img):
        new_img = cv2.resize(img, (self.size, self.size))
        return new_img if len(new_img.shape) == 3 else np.expand_dims(new_img, axis=2)


class AddMask(object):
    def __init__(self, parameters):
        self.mask_size = parameters['mask_size']

    def __call__(self, img, rnd_value):
        w, h, c = img.shape
        x, y = int(rnd_value[0] * (w - self.mask_size)), int(rnd_value[1] * (h - self.mask_size))
        img[x:x + self.mask_size, y:y + self.mask_size] = np.ones((self.mask_size, self.mask_size, c))
        return img


class AddIrregularMask(object):
    def __init__(self, parameters):
        self.mask_file_path = parameters['path']

    def __call__(self, img, rnd_value):
        mask_files = os.listdir(self.mask_file_path)
        idx = int(rnd_value[0] * len(mask_files))
        mask = cv2.imread(self.mask_file_path + mask_files[idx], cv2.IMREAD_UNCHANGED)
        w, h, c = img.shape
        mask = cv2.resize(mask, (w, h))
        mask[mask > 0] = 255
        mask = mask.reshape(w, h, 1) / 255.
        img = img + np.repeat(mask, c, axis=2)
        img = np.clip(img, 0, 1)
        return img


class AddBlindIrregularMask(object):
    def __init__(self, parameters):
        self.mask_file_path = parameters['path']

    def __call__(self, img, rnd_value):
        mask_files = os.listdir(self.mask_file_path)
        idx = int(rnd_value[0] * len(mask_files))
        mask = cv2.imread(self.mask_file_path + mask_files[idx], cv2.IMREAD_UNCHANGED)
        w, h, c = img.shape
        mask = cv2.resize(mask, (w, h))
        mask = mask.reshape(w, h, 1)
        mask = np.repeat(mask, c, axis=2)
        mask[mask > 0] = 255
        mask = mask / 255.
        if c == 1:
            return mask
        else:
            randdom_color = np.random.random((w, h, 3))
            img[mask == 1] = randdom_color[mask == 1]
            return img


class AddBlindColorIrregularMask(object):
    def __init__(self, parameters):
        self.mask_file_path = parameters['path']

    def __call__(self, img, rnd_value):
        mask_files = os.listdir(self.mask_file_path)
        idx = int(rnd_value[0] * len(mask_files))
        mask = cv2.imread(self.mask_file_path + mask_files[idx], cv2.IMREAD_UNCHANGED)
        w, h, c = img.shape
        mask = cv2.resize(mask, (w, h))
        mask = mask.reshape(w, h, 1)
        mask = np.repeat(mask, c, axis=2)
        mask[mask > 0] = 255
        mask = mask / 255.
        if c == 1:
            return mask
        else:
            r, g, b = np.random.uniform(0, 1, (3,))
            randdom_color = np.ones((w, h, c))
            randdom_color[:, :, 0] *= r
            randdom_color[:, :, 1] *= g
            randdom_color[:, :, 2] *= b
            img[mask == 1] = randdom_color[mask == 1]
            return img


class AddBlindCombinationIrregularMask(object):
    def __init__(self, parameters):
        self.mask_file_path = parameters['path']
        self.face_list = parameters['face_list']

    def __call__(self, img, rnd_value):
        mask_files = os.listdir(self.mask_file_path)
        idx = int(rnd_value[0] * len(mask_files))
        mask = cv2.imread(self.mask_file_path + mask_files[idx], cv2.IMREAD_UNCHANGED)
        w, h, c = img.shape
        mask = cv2.resize(mask, (w, h))
        mask = mask.reshape(w, h, 1)
        mask = np.repeat(mask, c, axis=2)
        mask[mask > 0] = 255
        mask = mask / 255.
        if c == 1:
            return mask
        else:
            face_idx = int(rnd_value[1] * len(self.face_list))
            face = cv2.imread(self.face_list[face_idx], cv2.IMREAD_UNCHANGED)
            face = cv2.resize(face, (w, h)) / 255.
            face[:, :, [0, 2]] = face[:, :, [2, 0]]  # BGR to RGB
            img[mask == 1] = face[mask == 1]
            return img


class AddBlindRandomIrregularMask(object):
    def __init__(self, parameters):
        self.mask_file_path = parameters['path']
        self.face_list = parameters['face_list']

    def __call__(self, img, rnd_value):
        mask_files = os.listdir(self.mask_file_path)
        idx = int(rnd_value[0] * len(mask_files))
        mask = cv2.imread(self.mask_file_path + mask_files[idx], cv2.IMREAD_UNCHANGED)
        w, h, c = img.shape
        mask = cv2.resize(mask, (w, h))
        mask = mask.reshape(w, h, 1)
        mask = np.repeat(mask, c, axis=2)
        mask[mask > 0] = 255
        mask = mask / 255.
        if c == 1:
            return mask
        else:
            typ = int(rnd_value[1] * 1000) % 3
            if typ == 0:
                randdom_cc = np.random.uniform(0, 1, (w, h, 3))
                img[mask == 1] = randdom_cc[mask == 1]
                return img
            elif typ == 1:
                face_idx = int(rnd_value[1] * len(self.face_list))
                face = cv2.imread(self.face_list[face_idx], cv2.IMREAD_UNCHANGED)
                face = cv2.resize(face, (w, h)) / 255.
                face[:, :, [0, 2]] = face[:, :, [2, 0]]  # BGR to RGB
                img[mask == 1] = face[mask == 1]
                return img
            else:
                r, g, b = np.random.uniform(0, 1, (3,))
                randdom_color = np.ones((w, h, c))
                randdom_color[:, :, 0] *= r
                randdom_color[:, :, 1] *= g
                randdom_color[:, :, 2] *= b
                img[mask == 1] = randdom_color[mask == 1]
                return img