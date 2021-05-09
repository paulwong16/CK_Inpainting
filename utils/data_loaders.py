import numpy as np
import cv2
import torch.utils.data.dataset
import utils.data_transforms


def collate_fn(batch):
    data = {}

    for sample in batch:
        _data = sample
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return data


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = {}
        gt_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) / 255.
        gt_img[:, :, [0, 2]] = gt_img[:, :, [2, 0]]  # BGR to RGB
        data['input'] = gt_img.copy()
        data['gt'] = gt_img.copy()

        if self.transforms is not None:
            data = self.transforms(data)

        return data


class AugDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = {}
        gt_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) / 255.
        gt_img[:, :, [0, 2]] = gt_img[:, :, [2, 0]]  # BGR to RGB
        data['input'] = gt_img.copy()
        data['aug1'] = gt_img.copy()
        data['aug2'] = gt_img.copy()
        data['gt'] = gt_img.copy()

        if self.transforms is not None:
            data = self.transforms(data)

        return data


class FFHQDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return Dataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.ffhq.mask_size
            },
            'objects': ['input']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt']
        }])


class FFHQAugDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return AugDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.ffhq.mask_size
            },
            'objects': ['input']
        }, {
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.ffhq.mask_size
            },
            'objects': ['aug1']
        }, {
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.ffhq.mask_size
            },
            'objects': ['aug2']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'aug1', 'aug2', 'gt']
        }])


class FFHQ1024DataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return Dataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'Resize',
            'parameters': {
                'size': 256
            },
            'objects': ['input', 'gt']
        }, {
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.ffhq.mask_size
            },
            'objects': ['input']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt']
        }])


class FFHQ1024MaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'Resize',
            'parameters': {
                'size': 256
            },
            'objects': ['input', 'gt', 'mask']
        }, {
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.ffhq.mask_size
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class MaskDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = {}
        gt_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) / 255.
        if len(gt_img.shape) != 3:
            w, h = gt_img.shape
            gt_img = gt_img.reshape((w, h, 1))
            gt_img = np.repeat(gt_img, 3, axis=2)
        gt_img[:, :, [0, 2]] = gt_img[:, :, [2, 0]]  # BGR to RGB
        data['input'] = gt_img.copy()
        data['gt'] = gt_img.copy()
        data['mask'] = np.zeros((gt_img.shape[0],gt_img.shape[1],1))

        if self.transforms is not None:
            data = self.transforms(data)

        return data


class FFHQMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.ffhq.mask_size
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQ1024IrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'Resize',
            'parameters': {
                'size': 256
            },
            'objects': ['input', 'gt', 'mask']
        }, {
            'callback': 'AddIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQ1024BlindIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'Resize',
            'parameters': {
                'size': 256
            },
            'objects': ['input', 'gt', 'mask']
        }, {
            'callback': 'AddBlindIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQ1024BlindCombIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        self.files_list = file_list
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'Resize',
            'parameters': {
                'size': 256
            },
            'objects': ['input', 'gt', 'mask']
        }, {
            'callback': 'AddBlindCombinationIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path,
                'face_list': self.files_list
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQ1024BlindRandomIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        self.files_list = file_list
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'Resize',
            'parameters': {
                'size': 256
            },
            'objects': ['input', 'gt', 'mask']
        }, {
            'callback': 'AddBlindRandomIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path,
                'face_list': self.files_list
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQBlindIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddBlindIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQBlindCombIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        self.files_list = file_list
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddBlindCombinationIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path,
                'face_list': self.files_list
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class FFHQBlindRandomIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.ffhq.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.ffhq.val_file_list
        else:
            file_list_path = self.cfg.ffhq.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.ffhq.base_path + file_list[i].strip('\n')
        self.files_list = file_list
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddBlindRandomIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path,
                'face_list': self.files_list
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class PlacesMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.places.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.places.val_file_list
        else:
            file_list_path = self.cfg.places.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.places.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddMask',
            'parameters': {
                'mask_size': self.cfg.places.mask_size
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


class PlacesIrregularMaskDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(subset)
        transforms = self._get_transforms()
        return MaskDataset(file_list, transforms)

    def _get_file_list(self, subset):
        if subset == 'train':
            file_list_path = self.cfg.places.train_file_list
        elif subset == 'val':
            file_list_path = self.cfg.places.val_file_list
        else:
            file_list_path = self.cfg.places.test_file_list
        with open(file_list_path, 'r') as f:
            file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = self.cfg.places.base_path + file_list[i].strip('\n')
        return file_list

    def _get_transforms(self):
        return utils.data_transforms.Compose([{
            'callback': 'AddIrregularMask',
            'parameters': {
                'path': self.cfg.mask_path
            },
            'objects': ['input', 'mask']
        }, {
            'callback': 'ToTensor',
            'objects': ['input', 'gt', 'mask']
        }])


DATASET_LOADER_MAPPING = {
    'FFHQ': FFHQDataLoader,
    'FFHQ1024': FFHQ1024DataLoader,
    'FFHQAug': FFHQAugDataLoader,
    'FFHQMask': FFHQMaskDataLoader,
    'FFHQ1024Mask': FFHQ1024MaskDataLoader,
    'FFHQIrregularMask': FFHQIrregularMaskDataLoader,
    'FFHQBlindIrregularMask': FFHQBlindIrregularMaskDataLoader,
    'FFHQBlindCombIrregularMask': FFHQBlindCombIrregularMaskDataLoader,
    'FFHQBlindRandomIrregularMask': FFHQBlindRandomIrregularMaskDataLoader,
    'FFHQ1024IrregularMask': FFHQ1024IrregularMaskDataLoader,
    'FFHQ1024BlindIrregularMask': FFHQ1024BlindIrregularMaskDataLoader,
    'FFHQ1024BlindCombIrregularMask': FFHQ1024BlindCombIrregularMaskDataLoader,
    'FFHQ1024BlindRandomIrregularMask': FFHQ1024BlindRandomIrregularMaskDataLoader,
    'PlacesMask': PlacesMaskDataLoader,
    'PlacesIrregularMask': PlacesIrregularMaskDataLoader,
}
