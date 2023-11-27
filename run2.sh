#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# 2 Otter fine tune AC Å® VI loss=context+query (short)
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--trained_ckpt="./log/AC_full_otter/batch128_epoch10_lr-5_yesno/final_weights.pt" \
--mimicit_path="./data/VI_full_jsons/VI_train_short_instructions.json" \
--images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
--train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
--val_mimicit_path="./data/VI_full_jsons/VI_val_short_instructions.json" \
--val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
--val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
--external_save_dir="./log/AC_VI_full_otter_short" \
--batch_size=128 \
--num_epochs=1 \
--include_context_loss="True" \
--ratio_extra_token_loss=1.0 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch1_lr-5_true \
--wandb_project=AC_VI_full_otter_short \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

# 2 Otter fine tune AC Å® VI loss=context+query (long)
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--trained_ckpt="./log/AC_full_otter/batch128_epoch10_lr-5_yesno/final_weights.pt" \
--mimicit_path="./data/VI_full_jsons/VI_train_long_instructions.json" \
--images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train.json" \
--train_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_train_pairs25_train.json" \
--val_mimicit_path="./data/VI_full_jsons/VI_val_long_instructions.json" \
--val_images_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val.json" \
--val_config_path="/home/data/MIMIC-IT/VI_full_jsons/VI_val_pairs1_train.json" \
--external_save_dir="./log/AC_VI_full_otter_long" \
--batch_size=128 \
--num_epochs=1 \
--include_context_loss="True" \
--ratio_extra_token_loss=1.0 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch1_lr-5_true \
--wandb_project=AC_VI_full_otter_long \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \