#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
--images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
--train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
--val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
--val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
--val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_pairs25_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_pairs1_train.json" \
--external_save_dir="./log/ACVI_loss_context_query" \
--batch_size=128 \
--num_epochs=3 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-5_pairs25_replacement-false_ACyesno \
--wandb_project=ACVI_loss_context_query \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
--images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
--train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
--val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
--val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
--val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_pairs5_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_pairs1_train.json" \
--external_save_dir="./log/ACVI_loss_context_query" \
--batch_size=128 \
--num_epochs=3 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-5_pairs5_replacement-false_ACyesno \
--wandb_project=ACVI_loss_context_query \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
--images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
--train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
--val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
--val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
--val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_pairs25_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_pairs1_train.json" \
--external_save_dir="./log/ACVI_loss_query" \
--batch_size=128 \
--num_epochs=3 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-5_pairs25_replacement-false_ACyesno \
--wandb_project=ACVI_loss_query \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--mimicit_path="/home/data/MIMIC-IT/AC/AC_train_instructions.json" \
--images_path="/home/data/MIMIC-IT/AC/AC_train.json" \
--train_config_path="/home/data/MIMIC-IT/AC/AC_train_train.json" \
--val_mimicit_path="/home/data/MIMIC-IT/AC/AC_val_instructions.json" \
--val_images_path="/home/data/MIMIC-IT/AC/AC_val.json" \
--val_config_path="/home/data/MIMIC-IT/AC/AC_val_train.json" \
--mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train.json" \
--train_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_train_pairs5_train.json" \
--val_mimicit_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_instructions.json" \
--val_images_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val.json" \
--val_config_ic_path="/home/data/MIMIC-IT/VI_jsons_yesno/VI_val_pairs1_train.json" \
--external_save_dir="./log/ACVI_loss_query" \
--batch_size=128 \
--num_epochs=3 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch3_lr-5_pairs5_replacement-false_ACyesno \
--wandb_project=ACVI_loss_query \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \