from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.ffhq = edict()
__C.ffhq.train_file_list = './dataset/ffhq/train_128.txt'
__C.ffhq.test_file_list = './dataset/ffhq/test_128.txt'
__C.ffhq.val_file_list = './dataset/ffhq/val_128.txt'
__C.ffhq.base_path = '../data/images1024x1024/'
__C.ffhq.mask_size = 128

__C.dataset = edict()
__C.dataset.train = 'FFHQ1024IrregularMask'
__C.dataset.test = 'FFHQ1024IrregularMask'

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