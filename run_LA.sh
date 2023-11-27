#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
--mimicit_ic_path="/home/data/MIMIC-IT/LA/LACR_I2I_instructions.json" \
--images_ic_path="/home/data/MIMIC-IT/LA/LA.json" \
--train_config_ic_path="/home/data/MIMIC-IT/LA/LACR_I2I_train.json" \
--external_save_dir="./log" \
--batch_size=4 \
--num_epochs=1 \
--report_to_wandb \
--wandb_entity=katlab_otter \
--run_name=LA_batch128 \
--wandb_project=OTTER-Image-MPT7B \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \