import logging
import torch

SEED = 1
cuda = torch.cuda.is_available()
"""
How to use logger? - Copy paste the below lines where ever logger is needed
from utils import logger_utils

logger = logger_utils.get_logger(__name__)

"""
optimizer_paras = {
  'lr':0.1,
  'momentum':0.9,
  'weight_decay':0.0   ## For L2 regularization
}

lr_scheduler_steplr_paras = {
  'step_size':12,
  'gamma':0.2
}

logger_config = {'log_filename':'logs/Session7_assignment',
                  'level': logging.INFO,
                  'format':'%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                  'datefmt':'%d-%m-%Y:%H:%M:%S'
                }

data = {
  'img_augs':{'random_rotation':{'angle_range': (-7.0, 7.0), 'fill':(1,1,1)},'horizontal_flip':{},'random_crop':{'size':32,'padding':4}},
   'normalize_paras':[(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
   'dataloader_args': dict(shuffle=True, batch_size=64, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4),
   'data_kind' : {"dataset_type":"open_datasets", "dataset_name": "CIFAR10", 'datasets_location':'data/datasets'},
}