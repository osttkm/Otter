#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
# --mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_instructions.json" \
# --images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train.json" \
# --train_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_pairs4_train.json" \
# --val_mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_instructions.json" \
# --val_images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val.json" \
# --val_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_pairs1_train.json" \
# --external_save_dir="./log/MI_init" \
# --batch_size=128 \
# --num_epochs=3 \
# --include_context_loss="True" \
# --ratio_extra_token_loss=0.2 \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch3_lr-5_pairs4_true \
# --wandb_project=MI_init \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
--mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_pairs4_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_pairs1_train.json" \
--external_save_dir="./log/MI_init" \
--batch_size=128 \
--num_epochs=3 \
--include_context_loss="False" \
--ratio_extra_token_loss=0.2 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-5_pairs4_false \
--wandb_project=MI_init \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
--mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_pairs4_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_pairs1_train.json" \
--external_save_dir="./log/MI_init" \
--batch_size=128 \
--num_epochs=3 \
--include_context_loss="False" \
--ratio_extra_token_loss=0.2 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-3_pairs4_false \
--wandb_project=MI_init \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-3 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
--mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_pairs4_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_pairs1_train.json" \
--external_save_dir="./log/MI_init" \
--batch_size=128 \
--num_epochs=3 \
--include_context_loss="False" \
--ratio_extra_token_loss=0.2 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-4_pairs4_false \
--wandb_project=MI_init \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-4 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
--mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_train_pairs4_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/MiniImagenet_jsons/MI_val_pairs1_train.json" \
--external_save_dir="./log/MI_init" \
--batch_size=128 \
--num_epochs=3 \
--include_context_loss="False" \
--ratio_extra_token_loss=0.2 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-6_pairs4_false \
--wandb_project=MI_init \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-6 \
--warmup_steps_ratio=0.01 \