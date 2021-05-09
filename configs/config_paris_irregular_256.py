from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.places = edict()
__C.places.train_file_list = './dataset/paris/train.txt'
__C.places.test_file_list = './dataset/paris/test.txt'
__C.places.val_file_list = './dataset/paris/test.txt'
__C.places.base_path = '../data/paris/'
__C.places.mask_size = 128

__C.dataset = edict()
__C.dataset.train = 'PlacesIrregularMask'
__C.dataset.test = 'PlacesIrregularMask'

__C.mask_path = '../data/mask/testing_mask_dataset/'

__C.const = edict()
__C.const.device = '0,1,2,3'
__C.const.num_workers = 8

__C.train = edict()
__C.train.batch_size = 12
__C.train.save_freq = 10
__C.train.epochs = 200
__C.train.init_lr = 1e-4
__C.train.lr_milestones = [40, 80, 120]
__C.train.gamma = 0.5
__C.train.betas = (.9, .999)


__C.dir = edict()
__C.dir.out_path = './output'