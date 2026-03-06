#!/bin/bash
gpuid=0 # Specifies the ID of a GPU used for training/evaluation
exp_name="exp"
ckpts="../../ckpts/Uni3D/model.pt"

cfgs=(
  "cfgs/finetune_scan_objbg.yaml"
  "cfgs/finetune_scan_objonly.yaml"
  "cfgs/finetune_scan_hardest.yaml"
  "cfgs/finetune_omniobject3d.yaml"
  "cfgs/finetune_3dgrocery100.yaml"
  "cfgs/finetune_mvpnet.yaml"
  "cfgs/finetune_objaverselvis.yaml"
  "cfgs/finetune_modelnet40.yaml"
  "cfgs/finetune_mcb_b.yaml"
  "cfgs/finetune_sh15_nonrigid.yaml"
  "cfgs/finetune_fg3d_airplane.yaml"
  "cfgs/finetune_fg3d_car.yaml"
  "cfgs/finetune_fg3d_chair.yaml"
)

for cfg in "${cfgs[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpuid python main.py --seed $RANDOM --config $cfg --finetune_model --exp_name $exp_name --ckpts $ckpts --stag_size 1
done

exit
