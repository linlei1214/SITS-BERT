import torch
from torch.utils.data import DataLoader
from model import SBERT
from trainer import SBERTFineTuner
from dataset import FinetuneDataset
import numpy as np
import random
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

def Config():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--file_path",
        default=None,
        type=str,
        required=False,
        help="The input data path.",
    )
    parser.add_argument(
        "--pretrain_path",
        default=None,
        type=str,
        required=False,
        help="The storage path of the pre-trained model.",
    )
    parser.add_argument(
        "--finetune_path",
        default=None,
        type=str,
        required=False,
        help="The output directory where the fine-tuning checkpoints will be written.",
    )
    parser.add_argument(
        "--valid_rate",
        default=0.03,
        type=float,
        help="")
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="The maximum length of input time series. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_features",
        default=10,
        type=int,
        help="",
    )
    parser.add_argument(
        "--num_classes",
        default=2,
        type=int,
        help="",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="",
    )
    parser.add_argument(
        "--layers",
        default=3,
        type=int,
        help="",
    )
    parser.add_argument(
        "--attn_heads",
        default=8,
        type=int,
        help="",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="",
    )
    return parser.parse_args()

if __name__ == "__main__":
    config = Config()

    train_file = config.file_path + 'Train.csv'
    valid_file = config.file_path + 'Validate.csv'
    test_file = config.file_path + 'Test.csv'

    print("Loading Data sets...")
    train_dataset = FinetuneDataset(train_file, config.num_features, config.max_length)
    valid_dataset = FinetuneDataset(valid_file, config.num_features, config.max_length)
    test_dataset = FinetuneDataset(test_file, config.num_features, config.max_length)
    print("training samples: %d, validation samples: %d, testing samples: %d" %
          (train_dataset.TS_num, valid_dataset.TS_num, test_dataset.TS_num))

    print("Creating Dataloader...")
    train_data_loader = DataLoader(train_dataset, shuffle=True,
                                   batch_size=config.batch_size, drop_last=False)
    valid_data_loader = DataLoader(valid_dataset, shuffle=False,
                                   batch_size=config.batch_size, drop_last=False)
    test_data_loader = DataLoader(test_dataset, shuffle=False,
                                  batch_size=config.batch_size, drop_last=True)


    print("Initialing SITS-BERT...")
    sbert = SBERT(config.num_features, hidden=config.hidden_size, n_layers=config.layers,
                  attn_heads=config.attn_heads, dropout=config.dropout)
    if config.pretrain_path is not None:
        print("Loading pre-trained model parameters...")
        sbert_path = config.pretrain_path + "checkpoint.bert.pth"
        sbert.load_state_dict(torch.load(sbert_path))

    print("Creating Downstream Task Trainer...")
    trainer = SBERTFineTuner(sbert, config.num_classes,
                             train_dataloader=train_data_loader,
                             valid_dataloader=valid_data_loader)

    print("Fine-tuning SITS-BERT...")
    OAAccuracy = 0
    for epoch in range(config.epochs):
        train_OA, _, valid_OA, _ = trainer.train(epoch)
        if OAAccuracy < valid_OA:
            OAAccuracy = valid_OA
            trainer.save(epoch, config.finetune_path)

    print("\n\n\n")
    print("Testing SITS-BERT...")
    trainer.load(config.finetune_path)
    OA, Kappa, AA, _ = trainer.test(test_data_loader)
    print('test_OA = %.2f, test_kappa = %.3f, test_AA = %.3f' % (OA, Kappa, AA))
