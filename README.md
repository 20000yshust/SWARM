# SWARM
[CVPR 2024] Not All Prompts Are Secure: A Switchable Backdoor Attack  Against Pre-trained Vision Transfomers

The naive version, you can simply use it through the instructions of visual prompt tuning.

I'm still working on cleaning the code now.


## Environment settings

See `vpt-main/env_setup.sh`


## Run SWARM

CUDA_VISIBLE_DEVICES=1 python tune_vtab.py --train-type "prompt" --config-file configs/prompt/cub.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "32" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "50" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.NAME "vtab-cifar(num_classes=100)" DATA.NUMBER_CLASSES "100" DATA.DATAPATH "your dataset path" MODEL.MODEL_ROOT "your checkpoint path" OUTPUT_DIR "your result path"
