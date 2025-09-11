#!/usr/bin/env python3

import os
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ms_depro import add_cfg
from ms_depro.data.datasets import builtin
from ms_depro.engine.train_loop import MSLTrainer
from ms_depro.modeling.meta_arch.rcnn import MSLCLIPRCNN
from ms_depro.modeling.backbone import build_clip_resnet_backbone
from ms_depro.modeling.proposal_generator.rpn import MSLRPN
from ms_depro.modeling.roi_heads.clip_roi_heads import CLIPRes5ROIHeads
from ms_depro.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        msl = cfg.TEST.MSL_MODE
        model_student = MSLTrainer.build_model(cfg)
        model_teacher = MSLTrainer.build_model(cfg)
        ts_model = EnsembleTSModel(model_teacher, model_student)

        DetectionCheckpointer(ts_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        assert msl in ["MSDA", "MSDG"]
        
        res = MSLTrainer.test(cfg, ts_model.modelTeacher if msl == "MSDA" else ts_model.modelStudent)    
        return res
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = MSLTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
