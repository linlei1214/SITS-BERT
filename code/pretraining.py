import torch
from dataset.dataset_wrapper import DataSetWrapper
from model import SBERT
from trainer import SBERTTrainer
import numpy as np
import random
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(123)

class Config(object):
    dataset_path = '../data/California-Unlabeled/California-10bands.csv'
    pretrain_path = '../checkpoints/pretrain/'
    valid_rate = 0.03
    max_length = 64
    num_features = 10
    epochs = 100
    batch_size = 512
    hidden_size = 256
    layers = 3
    attn_heads = 8
    learning_rate = 1e-4
    warmup_epochs = 10
    decay_gamma = 0.99
    dropout = 0.1
    gradient_clipping = 5.0

# def Config():
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument(
#         "--dataset_path",
#         default=None,
#         type=str,
#         required=False,
#         help="The input data path.",
#     )
#     parser.add_argument(
#         "--pretrain_path",
#         default=None,
#         type=str,
#         required=False,
#         help="The output directory where the pre-training checkpoints will be written.",
#     )
#     parser.add_argument(
#         "--valid_rate",
#         default=0.03,
#         type=float,
#         help="")
#     parser.add_argument(
#         "--max_length",
#         default=128,
#         type=int,
#         help="The maximum length of input time series. Sequences longer "
#         "than this will be truncated, sequences shorter will be padded.",
#     )
#     parser.add_argument(
#         "--num_features",
#         default=10,
#         type=int,
#         help="The dimensionality of satellite observations.",
#     )
#     parser.add_argument(
#         "--epochs",
#         default=100,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--batch_size",
#         default=64,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--hidden_size",
#         default=256,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--layers",
#         default=3,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--attn_heads",
#         default=8,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--learning_rate",
#         default=1e-4,
#         type=float,
#         help="",
#     )
#     parser.add_argument(
#         "--warmup_epochs",
#         default=100000,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--decay_gamma",
#         default=0.99,
#         type=float,
#         help="",
#     )
#     parser.add_argument(
#         "--dropout",
#         default=0.1,
#         type=float,
#         help="",
#     )
#     parser.add_argument(
#         "--gradient_clipping",
#         default=5.0,
#         type=float,
#         help="",
#     )
#     return parser.parse_args()
    
if __name__ == "__main__":
    config = Config()

    print("Loading training and validation data sets...")
    dataset = DataSetWrapper(batch_size=config.batch_size,
                             valid_size=config.valid_rate,
                             data_path=config.dataset_path,
                             num_features=config.num_features,
                             max_length=config.max_length)

    train_loader, valid_loader = dataset.get_data_loaders()

    print("Initialing SITS-BERT...")
    sbert = SBERT(config.num_features, hidden=config.hidden_size, n_layers=config.layers,
                  attn_heads=config.attn_heads, dropout=config.dropout)

    trainer = SBERTTrainer(sbert, config.num_features,
                           train_dataloader=train_loader,
                           valid_dataloader=valid_loader,
                           lr=config.learning_rate,
                           warmup_epochs=config.warmup_epochs,
                           decay_gamma=config.decay_gamma,
                           gradient_clipping_value=config.gradient_clipping)

    print("Pre-training SITS-BERT...")
    mini_loss = np.Inf
    for epoch in range(config.epochs):
        train_loss, valida_loss = trainer.train(epoch)
        if mini_loss > valida_loss:
            mini_loss = valida_loss
            trainer.save(epoch, config.pretrain_path)

