#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/AC_batch128_epoch5_lr-5_yesno" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_pairs25_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_pairs1_train.json" \
--external_save_dir="./log/AC-VI_loss_context_query" \
--batch_size=128 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch1_lr-5_pairs25_weight5_ACyesno \
--wandb_project=AC-VI_loss_context_query \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/AC_batch128_epoch5_lr-5_yesno" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_pairs5_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_pairs1_train.json" \
--external_save_dir="./log/AC-VI_loss_context_query" \
--batch_size=128 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch1_lr-5_pairs5_weight5_ACyesno \
--wandb_project=AC-VI_loss_context_query \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/AC_batch128_epoch5_lr-4" \
# --mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_train_instructions.json" \
# --images_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_train.json" \
# --train_config_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_val_instructions.json" \
# --val_images_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_val.json" \
# --val_config_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/AC-VI_loss_context_query" \
# --batch_size=128 \
# --num_epochs=1 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-5_pairs25_weight4 \
# --wandb_project=AC-VI_loss_context_query \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/AC_batch128_epoch5_lr-4" \
# --mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_train_instructions.json" \
# --images_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_train.json" \
# --train_config_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_train_pairs25_train.json" \
# --val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_val_instructions.json" \
# --val_images_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_val.json" \
# --val_config_ic_path="/home/data/MIMIC-IT/VI_jsons/VI_val_pairs1_train.json" \
# --external_save_dir="./log/AC-VI_loss_context_query" \
# --batch_size=128 \
# --num_epochs=1 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-4_pairs25_weight4 \
# --wandb_project=AC-VI_loss_context_query \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-4 \
# --warmup_steps_ratio=0.01 \