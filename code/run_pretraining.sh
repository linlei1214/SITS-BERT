export PATH=/data02/linlei/programFiles/anaconda3/envs/sits_bert_envs/bin/:$PATH
export CUDA_VISIBLE_DEVICES=0
python pretraining.py \
    --dataset_path '../data/California-Unlabeled/California-10bands.csv' \
    --pretrain_path '../checkpoints/pretrain_tmp/' \
    --valid_rate 0.03 \
    --max_length 64 \
    --num_features 10 \
    --epochs 100 \
    --batch_size 256 \
    --hidden_size 256 \
    --layers 3 \
    --attn_heads 8 \
    --learning_rate 1e-4 \
    --warmup_epochs 10 \
    --decay_gamma 0.99 \
    --dropout 0.1 \
    --gradient_clipping 5.0






