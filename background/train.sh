export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="../data/background/train"
export OUTPUT_DIR="../logs/background"

accelerate launch train_background.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--train_text_encoder \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--resolution=512 \
--train_batch_size=8 \
--gradient_checkpointing \
--learning_rate=1e-5 \
--lr_scheduler="constant" \
--lr_warmup_steps=500 \
--num_train_epochs=50 \
--with_spatial_loss \
--checkpointing_steps=10000
