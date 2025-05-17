import argparse
import os
import time
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import inference
from model import build_segmenter
from utils.dataset import RefDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.output_dir = args.output_folder
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)
        
    # Set device for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(device)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build dataset & dataloader
    test_data = RefDataset(lmdb_dir=args.test_lmdb,
                           mask_dir=args.mask_root,
                           dataset=args.dataset,
                           split=args.test_split,
                           mode='test',
                           input_size=args.input_size,
                           word_length=args.word_len)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=10,
                                              shuffle=False,
                                              num_workers=24,
                                              pin_memory=True)

    # build model
    model, _ = build_segmenter(args)
    model = model.to(device) 

    args.model_dir = os.path.join(args.output_dir, "last_model.pth")
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    # inference with FPS measurement
    start_time = time.time()
    mean_iou, _ = inference(test_loader, model, args)
    elapsed = time.time() - start_time
    num_batches = len(test_loader)
    fps = num_batches / elapsed if elapsed > 0 else float('inf')
    logger.info(f"Average FPS: {fps:.2f}")
    print(f"Average FPS: {fps:.2f}")


if __name__ == '__main__':
    main()
