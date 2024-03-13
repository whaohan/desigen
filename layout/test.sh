EXP=swin
DATASET=webui
COMMAND=category_generate
python main.py --encode_backbone swin --encode_embd 1024 \
--dataset $DATASET --exp $EXP --evaluate \
--decoder_path ../logs/$DATASET/$EXP/checkpoints/decoder.pth \
--encoder_path ../logs/$DATASET/$EXP/checkpoints/encoder.pth \
--eval_command $COMMAND \
--calculate_harmony \
--save_pkl 

python eval.py webui logs/$DATASET/$EXP/generated_layout_$COMMAND.pth