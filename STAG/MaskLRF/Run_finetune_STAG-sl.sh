#!/bin/bash
gpuid=0 # Specifies the ID of a GPU used for training/evaluation
ckpts="../../ckpts/MaskLRF/ckpt-last.pth"
exp_name_base="exp"
myarg_batch_size=32
myarg_rot_train="nr" # nr (no rotation) or so3 (random rotation)
myarg_rot_test="so3"
myarg_lrf_axis1="pca"
myarg_lrf_axis2="mean"
myarg_attn_mode="rel_contextual"
myarg_attn_subsample_rate=0.125 # k=16
myarg_attn_target="knn_dilate"
vote_flag="" # no voting

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
  dataset=${cfg#cfgs/finetune_}; dataset=${dataset%.yaml}
  exp_name=$exp_name_base\_$dataset

  CUDA_VISIBLE_DEVICES=$gpuid python -u main.py --seed $RANDOM --config $cfg --finetune_model --ckpts $ckpts --exp_name $exp_name $vote_flag \
                                             --myarg_batch_size $myarg_batch_size \
                                             --myarg_rot_train $myarg_rot_train --myarg_rot_test $myarg_rot_test \
                                             --myarg_lrf_axis1 $myarg_lrf_axis1 --myarg_lrf_axis2 $myarg_lrf_axis2 \
                                             --myarg_attn_mode $myarg_attn_mode --myarg_attn_subsample_rate $myarg_attn_subsample_rate \
                                             --myarg_attn_target $myarg_attn_target \
                                             --stag_size 1
done

exit
