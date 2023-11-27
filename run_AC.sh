#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}


# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
# --mimicit_path="/home/dataset/yyama_dataset/tasks/AC/AC_train_instructions.json" \
# --images_path="/home/dataset/yyama_dataset/tasks/AC/AC_train.json" \
# --train_config_path="/home/dataset/yyama_dataset/tasks/AC/AC_train_train.json" \
# --external_save_dir="./log/AC_newtext" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=oshita_otter \
# --run_name=batch128_epoch5_lr-5_AC_new_text \
# --wandb_project=AC_full \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \


# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
# --mimicit_path="/home/dataset/yyama_dataset/tasks/AC/default_AC_train_instructions.json" \
# --images_path="/home/dataset/yyama_dataset/tasks/AC/default_AC_train.json" \
# --train_config_path="/home/dataset/yyama_dataset/AC/default_AC_train_train.json" \
# --external_save_dir="./log/AC_previous" \
# --batch_size=128 \
# --num_epochs=5 \
# --report_to_wandb \
# --wandb_entity=oshita_otter \
# --run_name=batch128_epoch5_lr-5_AC_previous_text \
# --wandb_project=AC_full \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \


# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
# --mimicit_path="/home/dataset/yyama_dataset/tasks/AC/AC_train_instructions.json" \
# --images_path="/home/dataset/yyama_dataset/tasks/AC/AC_train.json" \
# --train_config_path="/home/dataset/yyama_dataset/tasks/AC/AC_train_train.json" \
# --external_save_dir="./log/AC_newtext" \
# --batch_size=128 \
# --num_epochs=10 \
# --report_to_wandb \
# --wandb_entity=oshita_otter \
# --run_name=batch128_epoch10_lr-5_AC_new_text \
# --wandb_project=AC_full \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \


accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-MPT7B-Init" \
--mimicit_path="/home/dataset/yyama_dataset/tasks/AC/default_AC_train_instructions.json" \
--images_path="/home/dataset/yyama_dataset/tasks/AC/default_AC_train.json" \
--train_config_path="/home/dataset/yyama_dataset/tasks/AC/default_AC_train_train.json" \
--external_save_dir="./log/AC_previous" \
--batch_size=128 \
--num_epochs=10 \
--report_to_wandb \
--wandb_entity=oshita_otter \
--run_name=batch128_epoch10_lr-5_AC_previous_text \
--wandb_project=AC_full \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \


# python pipeline/demo/evaluate_AC.py \
# python pipeline/demo/evaluate_AC_MVTec.py \
