from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from ..model import SBERT, SBERTPrediction
from model import SBERT, SBERTPrediction

torch.manual_seed(123)

class SBERTTrainer:

    def __init__(self, sbert: SBERT, num_features: int,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float = 1e-4, warmup_epochs: int = 10, decay_gamma: float = 0.99,
                 with_cuda: bool = True, cuda_devices=None,
                 log_freq: int = 10, gradient_clipping_value = 5.0):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.sbert = sbert
        self.model = SBERTPrediction(sbert, num_features).to(self.device)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for model pre-training" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.warmup_epochs = warmup_epochs
        self.optim_schedule = lr_scheduler.ExponentialLR(self.optim, gamma=decay_gamma)
        self.gradient_clippling = gradient_clipping_value
        self.criterion = nn.MSELoss(reduction='none')

        self.writer = SummaryWriter()
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):

        data_iter = tqdm(enumerate(self.train_dataloader),
                         desc="EP_%s:%d" % ("train", epoch),
                         total=len(self.train_dataloader),
                         bar_format="{l_bar}{r_bar}")

        train_loss = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            mask_prediction = self.model(data["bert_input"].float(),
                                         data["time"].long(),
                                         data["bert_mask"].long())

            loss = self.criterion(mask_prediction, data["bert_target"].float())
            mask = data["loss_mask"].unsqueeze(-1)
            loss = (loss * mask.float()).sum() / mask.sum()

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clippling)
            self.optim.step()

            train_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": train_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        train_loss = train_loss / len(data_iter)
        self.writer.add_scalar('train_loss', train_loss, global_step=epoch)

        valid_loss = self._validate()
        self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)

        if epoch >= self.warmup_epochs:
            self.optim_schedule.step()
        self.writer.add_scalar('cosine_lr_decay', self.optim_schedule.get_lr()[0], global_step=epoch)

        print("EP%d, train_loss=%.5f, validate_loss=%.5f" % (epoch, train_loss, valid_loss))
        return train_loss, valid_loss

    def _validate(self):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for data in self.valid_dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}

                mask_prediction = self.model(data["bert_input"].float(),
                                             data["time"].long(),
                                             data["bert_mask"].long())

                loss = self.criterion(mask_prediction, data["bert_target"].float())
                mask = data["loss_mask"].unsqueeze(-1)
                loss = (loss * mask.float()).sum() / mask.sum()

                valid_loss += loss.item()
                counter += 1

            valid_loss /= counter

        self.model.train()
        return valid_loss

    def save(self, epoch, file_path):
        output_path = file_path + "checkpoint.tar"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }, output_path)

        bert_path = file_path + "checkpoint.bert.pth"
        torch.save(self.sbert.state_dict(), bert_path)
        self.sbert.to(self.device)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, epoch, file_path):
        input_path = file_path + "checkpoint.tar"

        checkpoint = torch.load(input_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.train()

        print("EP:%d Model loaded from:" % epoch, input_path)
        return input_path


