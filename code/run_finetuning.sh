#export CUDA_VISIBLE_DEVICES=1
python finetuning.py \
    --file_path '../data/California-Labeled/' \
    --pretrain_path '../checkpoints/pretrain/' \
    --finetune_path '../checkpoints/finetune/' \
    --max_length 64 \
    --num_features 10 \
    --num_classes 13 \
    --epochs 100 \
    --batch_size 128 \
    --hidden_size 256 \
    --layers 3 \
    --attn_heads 8 \
    --learning_rate 2e-4 \
    --dropout 0.1

