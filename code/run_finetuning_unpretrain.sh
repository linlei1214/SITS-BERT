export PATH=/data02/linlei/programFiles/anaconda3/envs/sits_bert_envs/bin/:$PATH
export CUDA_VISIBLE_DEVICES=2
python finetuning.py \
    --file_path '../data/California-Labeled/' \
    --finetune_path '../checkpoints/unpretrain/' \
    --max_length 64 \
    --num_features 10 \
    --num_classes 13 \
    --epochs 300 \
    --batch_size 128 \
    --hidden_size 256 \
    --layers 3 \
    --attn_heads 8 \
    --learning_rate 2e-4 \
    --dropout 0.1

