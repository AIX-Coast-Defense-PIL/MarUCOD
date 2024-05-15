import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model.train import LitModel
from model.utils import ModelExporter, load_weights
from model.wodis import WODIS_model
from data.data_loader import MaSTr1325Dataset
from data.transforms import get_augmentation_transform, PytorchHubNormalization, Normalization_B

import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = './data/images'
IMG_SIZE = [1024, 1280]
DEVICE_BATCH_SIZE = 8
NUM_CLASSES = 3
PATIENCE = None
LOG_STEPS = 20
NUM_WORKERS = 4
NUM_GPUS = -1 # All visible GPUs
RANDOM_SEED = None
OUTPUT_DIR = 'seg/output'
PRETRAINED_DEEPLAB = True
PRECISION = 32
MONITOR_VAR = 'val/iou/obstacle'
MONITOR_VAR_MODE = 'max'
NORMALIZE_B = None # True or None
SHIFT_B = None


def get_arguments(input_args=None):
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WODIS Network Train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", type=str, default=DATASET_DIR, help="Path to the training dataset directory.")
    parser.add_argument("--add_source", type=str, default=None, help="Path to the training dataset directory.")
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=IMG_SIZE, help='inference size h,w')
    parser.add_argument("--batch_size", type=int, default=DEVICE_BATCH_SIZE, help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--validation", action="store_true", help="Report performance on validation set and use early stopping.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background).")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Patience for early stopping (how many epochs to wait without increase).")
    parser.add_argument("--log_steps", type=int, default=LOG_STEPS, help="Number of steps between logging variables.")
    parser.add_argument("--gpus", default=NUM_GPUS, help="Number of gpus (or GPU ids) used for training.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of workers used for data loading.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results.")
    parser.add_argument("--pretrained", type=bool, default=PRETRAINED_DEEPLAB, help="Use pretrained DeepLab weights.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="SubDirectory where the output will be stored (models and logs)")
    parser.add_argument("--datetime", type=str, default='0000', help="Directory where the output will be stored (models and logs)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model. Used to create model and log directories inside the output directory.")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to the pretrained weights to be used.")
    parser.add_argument("--monitor_metric", type=str, default=MONITOR_VAR, help="Validation metric to monitor for early stopping and best model saving.")
    parser.add_argument("--monitor_metric_mode", type=str, default=MONITOR_VAR_MODE, choices=['min', 'max'], help="Maximize or minimize the monitored metric.")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable on-the-fly image augmentation of the dataset.")
    parser.add_argument("--precision", default=PRECISION, type=int, choices=[16,32], help="Floating point precision.")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from specified checkpoint.")
    parser.add_argument("--data_sampling", default=None, type=int, choices=[0, 2, 4, 8, 16], help="Train dataset random sampling.")
    parser.add_argument("--normalize_t", action="store_true", help="ImageNet Normalization")
    parser.add_argument("--normalize_b", action="store_true", help="Brightness Normalization")
    parser.add_argument("--shift_b", action="store_true", help="Brightness Shift")

    parser = LitModel.add_argparse_args(parser)

    args = parser.parse_args(input_args)

    return args

def train_wodis(args):
    # Use or create random seed
    args.random_seed = pl.seed_everything(args.random_seed)

    normalize_t = PytorchHubNormalization() if args.normalize_t else None
    NORMALIZE_B = Normalization_B(mean_=0.6814, std_=0.2048) if args.normalize_b else None 
    ''' mastr1325 mean: 0.681, mastr1478 mean: 0.661, LaRS mean: 0.5244, 
        mastr1325 std: 0.2042, mastr1478 std: 0.226, LaRS std: 0.241 '''
    SHIFT_B = True if args.shift_b else None
    transform = None

    # Data augmentation
    if not args.no_augmentation:
        transform = get_augmentation_transform()

    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    train_ds = MaSTr1325Dataset(args.source, transform=transform,
                                normalize_t=normalize_t, normalize_b = NORMALIZE_B, shift_b = SHIFT_B, yolo_resize=(args.imgsz, 32))
    
    if args.add_source:
        add_ds = MaSTr1325Dataset(args.add_source, transform=transform,
                                    normalize_t=normalize_t, normalize_b = NORMALIZE_B, shift_b = SHIFT_B, yolo_resize=(args.imgsz, 32))
        train_ds = ConcatDataset([train_ds, add_ds])

    if args.data_sampling:
        train_size = len(train_ds)//args.data_sampling
        val_size = len(train_ds) - train_size
        train_ds, _ = random_split(train_ds, [train_size, val_size], torch.Generator().manual_seed(42))
    
    val_dl = None
    if args.validation:
        train_size = int(len(train_ds)*0.8)
        val_size = len(train_ds) - train_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size], torch.Generator().manual_seed(args.random_seed))
        val_ds.transform, val_ds.normalize_b, val_ds.include_original = None, None, True
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True)
    print(f"length of the train & val dataset: ({len(train_dl)}, {len(val_dl)})")
    
    model = WODIS_model(num_classes=args.num_classes)

    if args.pretrained_weights is not None:
        print(f"Loading weights from: {args.pretrained_weights}")
        state_dict = load_weights(args.pretrained_weights)
        model.load_state_dict(state_dict)

    model = LitModel(model, args.num_classes, args)

    logs_path = os.path.join(args.output_dir, 'logs')
    logger = pl_loggers.TensorBoardLogger(logs_path, args.model_name, version=args.datetime)
    logger.log_hyperparams(args)

    callbacks = []

    if args.validation:
        # Val: Early stopping and best model saving
        if args.patience is not None:
            callbacks.append(EarlyStopping(monitor=args.monitor_metric, patience=args.patience, mode=args.monitor_metric_mode))
        # callbacks.append(ModelCheckpoint(save_last=True, save_top_k=1, monitor=args.monitor_metric, mode=args.monitor_metric_mode))
        callbacks.append(ModelCheckpoint(save_top_k=1, save_weights_only=True, #filename='',
                                         monitor=args.monitor_metric, mode=args.monitor_metric_mode))

        callbacks.append(ModelExporter())

    print('The number of train datasets: ', len(train_dl))
    print('The number of validation datasets: ', len(val_dl))
    print('The number of epochs: ', args.epochs)

    trainer = pl.Trainer(logger=logger,
                         gpus=args.gpus,
                         max_epochs=args.epochs,
                         accelerator='ddp',
                         resume_from_checkpoint=args.resume_from,
                         callbacks=callbacks,
                         sync_batchnorm=True,
                         log_every_n_steps=args.log_steps,
                         precision=args.precision)
    trainer.fit(model, train_dl, val_dl)
    print('Train Finish!')


def main():
    args = get_arguments()
    print(args)

    train_wodis(args)


if __name__ == '__main__':
    main()
