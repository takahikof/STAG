## Pre-trained parameters

### Point-MAE
Please download the file "pretrain.pth" from [Point-MAE repo](https://github.com/Pang-Yatian/Point-MAE).
### MaskLRF
Please download the file "ckpt-last.pth" from [MaskLRF repo](https://github.com/takahikof/MaskLRF).
### Uni3D-S
Please download the file "model.pt" from [Uni3D repo](https://huggingface.co/BAAI/Uni3D/tree/main/modelzoo/uni3d-s).<br>

The downloaded parameter files should be placed under the "ckpts" directory.<br>
The overall directory structure should be:
```
├──FullFinetuning/
├──STAG/
├──ckpts/
│   ├──Point-MAE/
│      ├──pretrain.pth
│   ├──MaskLRF/
│      ├──ckpt-last.pth
│   ├──Uni3D/
│      ├──model.pt
├──data/
├──prepare/
```
I'd like to express my sincere gratitude to all those who devoted their efforts to developing and releasing each pretrained model.

