#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/home/data/MIMICIT/LA/LACR_T2T_instructions.json" \
# --images_ic_path="/home/data/MIMICIT/LA/LA.json" \
# --train_config_ic_path="/home/data/MIMICIT/LA/LACR_T2T_train.json" \
# --external_save_dir="./log/MIMIC_IT" \
# --batch_size=4 \
# --num_epochs=6 \
# --gradient_accumulation_steps=1 \
# --backward_timing="separate" \
# --save_ckpt_each_epoch \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch4_epoch6_lr-5 \
# --wandb_project=MIMIC_IT \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 初期重みから a6000 (LACONV、DC、E4D、VSTなし)
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
--mimicit_path="/home/data/MIMICIT/CGD/CGD_instructions.json","/home/data/MIMICIT/SD/SD_instructions.json" \
--images_path="/home/data/MIMICIT/CGD/CGD.json","/home/data/MIMICIT/SD/SD.json" \
--train_config_path="/home/data/MIMICIT/CGD/CGD_train.json","/home/data/MIMICIT/SD/SD_train.json" \
--mimicit_ic_path="/home/data/MIMICIT/LA/LACR_T2T_instructions.json","/home/data/MIMICIT/LA/LACR_I2I_instructions.json" \
--images_ic_path="/home/data/MIMICIT/LA/LA.json","/home/data/MIMICIT/LA/LA.json" \
--train_config_ic_path="/home/data/MIMICIT/LA/LACR_T2T_train.json","/home/data/MIMICIT/LA/LACR_I2I_train.json" \
--mimicit_vt_path="/home/data/MIMICIT/SN/SN_instructions.json","/home/data/MIMICIT/TVC/TVC_instructions.json" \
--images_vt_path="/home/data/MIMICIT/SN/SN.json","/home/data/MIMICIT/TVC/TVC.json" \
--external_save_dir="./log/MIMIC_IT" \
--batch_size=4 \
--num_epochs=6 \
--gradient_accumulation_steps=1 \
--backward_timing="separate" \
--save_ckpt_each_epoch \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch4_epoch6_lr-5 \
--wandb_project=MIMIC_IT \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
# --mimicit_path="/home/data/MIMICIT/CGD/CGD_instructions.json","/home/data/MIMICIT/SD/SD_instructions.json" \
# --images_path="/home/data/MIMICIT/CGD/CGD.json","/home/data/MIMICIT/SD/SD.json" \
# --train_config_path="/home/data/MIMICIT/CGD/CGD_train.json","/home/data/MIMICIT/SD/SD_train.json" \
# --mimicit_ic_path="/home/data/MIMICIT/LA/LACR_I2I_instructions.json","/home/data/MIMICIT/LA/LACONV_instructions.json","/home/data/MIMICIT/LA/LACR_I2I_instructions.json" \
# --images_ic_path="/home/data/MIMICIT/LA/LA.json","/home/data/MIMICIT/LA/LA.json","/home/data/MIMICIT/LA/LA.json" \
# --train_config_ic_path="/home/data/MIMICIT/LA/LACR_I2I_train.json","/home/data/MIMICIT/LA/LACONV_train.json","/home/data/MIMICIT/LA/LACR_I2I_train.json" \
# --mimicit_vt_path="/home/data/MIMICIT/E4D/E4D_instructions.json","/home/data/MIMICIT/SN/SN_instructions.json","/home/data/MIMICIT/TVC/TVC_instructions.json" \
# --images_vt_path="/home/data/MIMICIT/E4D/E4D.json","/home/data/MIMICIT/SN/SN.json","/home/data/MIMICIT/TVC/TVC.json" \
# --external_save_dir="./log/MIMIC_IT" \
# --batch_size=4 \
# --num_epochs=6 \
# --gradient_accumulation_steps=1 \
# --backward_timing="separate" \
# --save_ckpt_each_epoch \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch4_epoch6_lr-5 \
# --wandb_project=MIMIC_IT \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \


# LAのT2Tって？
# VST動く？
# １つずつshape確認
# new, pastでなんとか？