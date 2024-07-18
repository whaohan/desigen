python main.py --dataset webui --exp layout \
--data_dir ../data \
--epoch 100 --lr 1.5e-5 --lr_decay \
--encode_backbone swin --encode_embd 1024 \
--finetune_vb --pretrain_vb
