#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# # 1 DC+ICL separate
# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
# --mimicit_path="./data/abst/jsons/Defect_Classification/defect_name/train_instructions.json" \
# --images_path="./data/abst/jsons/Defect_Classification/defect_name/train_images.json" \
# --train_config_path="./data/abst/jsons/Defect_Classification/defect_name/train_train.json" \
# --mimicit_ic_path="./data/abst/jsons/ICL/seihin_defect_name+long_QA/train_instructions.json" \
# --images_ic_path="./data/abst/jsons/ICL/seihin_defect_name+long_QA/train_images.json" \
# --train_config_ic_path="./data/abst/jsons/ICL/seihin_defect_name+long_QA/train_pairs25_train.json" \
# --external_save_dir="./log/DCICL" \
# --batch_size=128 \
# --num_epochs=1 \
# --gradient_accumulation_steps=1 \
# --backward_timing="separate" \
# --report_to_wandb \
# --wandb_entity=katlab_otter \
# --run_name=batch128_epoch1_lr-5_separate \
# --wandb_project=DCICL \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \

# 1 ICL フルデータ abst時と同じプロンプト
accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="./data/full/jsons/ICL/seihin_defect_name+longQA/train_instructions.json" \
--images_ic_path="./data/full/jsons/ICL/seihin_defect_name+longQA/train_images.json" \
--train_config_ic_path="./data/full/jsons/ICL/seihin_defect_name+longQA/train_pairs25_train.json" \
--external_save_dir="./log/ICL_full_abstprompt" \
--batch_size=64 \
--num_epochs=1 \
--gradient_accumulation_steps=2 \
--backward_timing="separate" \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=batch128_epoch1_lr-5_separate \
--wandb_project=ICL_full_abstprompt \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \