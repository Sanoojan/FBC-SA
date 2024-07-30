import argparse
import torch
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# new imports
from yacs.config import CfgNode as CN
import copy

# datasets
import datasets.ssdg_pacs
import datasets.ssdg_officehome
import os
import wandb
import random

wandb.login(key="fa0767adc156a87ed43a394680774f3116fc3ed2")

# trainers
import trainers.FBCSA
import trainers.FBCSA_UP

os.environ["WANDB_DISABLED"] = "true"

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    # breakpoint()
    cfg.TRAINER.FBCSA_UP = CN()
    cfg.TRAINER.FBCSA_UP.CONF_THRE = 0.95  # confidence threshold
    cfg.TRAINER.FBCSA_UP.STRONG_TRANSFORMS = ()  # strong augmentations
    cfg.TRAINER.FBCSA_UP.C_OPTIM = copy.deepcopy(cfg.OPTIM)  # classifier's optim setting
    cfg.TRAINER.FBCSA_UP.CLASSIFIER = "normal"  # stochastic or normal
    # pass
    cfg.TRAINER.FBCSA = CN()
    cfg.TRAINER.FBCSA.CONF_THRE = 0.95  # confidence threshold
    cfg.TRAINER.FBCSA.STRONG_TRANSFORMS = ()  # strong augmentations
    cfg.TRAINER.FBCSA.C_OPTIM = copy.deepcopy(cfg.OPTIM)  # classifier's optim setting
    cfg.TRAINER.FBCSA.CLASSIFIER = "normal"  # stochastic or normal
    
    cfg.TRAINER.FBCSA_dom_align = CN()
    cfg.TRAINER.FBCSA_dom_align.CONF_THRE = 0.95  # confidence threshold
    cfg.TRAINER.FBCSA_dom_align.STRONG_TRANSFORMS = ()  # strong augmentations
    cfg.TRAINER.FBCSA_dom_align.C_OPTIM = copy.deepcopy(cfg.OPTIM)  # classifier's optim setting
    cfg.TRAINER.FBCSA_dom_align.CLASSIFIER = "normal"  # stochastic or normal
    
    cfg.TRAINER.FBCSA_NO_CLASSIFIER = CN()
    cfg.TRAINER.FBCSA_NO_CLASSIFIER.CONF_THRE = 0.95  # confidence threshold
    cfg.TRAINER.FBCSA_NO_CLASSIFIER.STRONG_TRANSFORMS = ()  # strong augmentations
    cfg.TRAINER.FBCSA_NO_CLASSIFIER.C_OPTIM = copy.deepcopy(cfg.OPTIM)  # classifier's optim setting
    cfg.TRAINER.FBCSA_NO_CLASSIFIER.CLASSIFIER = "normal"  # stochastic or normal
    
    cfg.TRAINER.FIXMATCH = CN()
    cfg.TRAINER.FIXMATCH.CONF_THRE = 0.95  # confidence threshold
    cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS = ()  # strong augmentations
    cfg.TRAINER.FIXMATCH.C_OPTIM = copy.deepcopy(cfg.OPTIM)  # classifier's optim setting
    cfg.TRAINER.FIXMATCH.CLASSIFIER = "normal"  # stochastic or normal
    # cfg.TRAINER.FIXMATCH.EXP.CONFIDENCE_WEIGHTAGE= False # confidence weightage
    

def setup_cfg(args):
    # breakpoint()
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    # breakpoint()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    
    save_location = args.output_dir
    # check if log.txt exists
    if os.path.exists(save_location + "/log.txt"):
        read_file = open(save_location + "/log.txt", "r")
        # check if the last line contains "elapsed"
        lines = read_file.readlines()
        if "Elapsed" in lines[-1]:
            print("Skipping as already trained")
            return   
        else:
            # delete the log file
            os.remove(save_location + "/log.txt")
    
        
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    nowname=args.trainer + "_"+args.exp_name + args.dataset_config_file.split("/")[-1].split(".")[0] +"_seed_" +str(args.seed)+"_domain_"+args.target_domains[0]
    wandb.init(
    # set the wandb project where this run will be logged
        project="FBCSA_extension",
        name=nowname,
        # track hyperparameters and run metadata
        
        config=cfg)
    
    
    
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--exp-name", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
