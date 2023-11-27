#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# 初期重みから a100 (DC、VSTなし、勾配ためる)
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
--mimicit_path="/home/data/MIMICIT/CGD/CGD_instructions.json","/home/data/MIMICIT/SD/SD_instructions.json" \
--images_path="/home/data/MIMICIT/CGD/CGD.json","/home/data/MIMICIT/SD/SD.json" \
--train_config_path="/home/data/MIMICIT/CGD/CGD_train.json","/home/data/MIMICIT/SD/SD_train.json" \
--mimicit_ic_path="/home/data/MIMICIT/LA/LACR_T2T_instructions.json","/home/data/MIMICIT/LA/LACR_I2I_instructions.json","/home/data/MIMICIT/LA/LACONV_instructions.json" \
--images_ic_path="/home/data/MIMICIT/LA/LA.json","/home/data/MIMICIT/LA/LA.json","/home/data/MIMICIT/LA/LA.json" \
--train_config_ic_path="/home/data/MIMICIT/LA/LACR_T2T_train.json","/home/data/MIMICIT/LA/LACR_I2I_train.json","/home/data/MIMICIT/LA/LACONV_train.json" \
--mimicit_vt_path="/home/data/MIMICIT/E4D/E4D_instructions.json","/home/data/MIMICIT/SN/SN_instructions.json","/home/data/MIMICIT/TVC/TVC_instructions.json" \
--images_vt_path="/home/data/MIMICIT/E4D/E4D.json","/home/data/MIMICIT/SN/SN.json","/home/data/MIMICIT/TVC/TVC.json" \
--external_save_dir="./log/MIMIC_IT" \
--batch_size=1 \
--num_epochs=6 \
--gradient_accumulation_steps=4 \
--backward_timing="separate" \
--save_ckpt_each_epoch \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch4_epoch6_lr-5_a100 \
--wandb_project=MIMIC_IT \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \