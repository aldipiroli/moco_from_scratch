import copy

import torch
from tqdm import tqdm
from utils.augmentations import MoCoAugmentations
from utils.trainer_base import TrainerBase


class MoCoKeyQueue:
    def __init__(self, max_batches, batch_size):
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.q = []

    def insert_batch(self, x):
        self.q.extend(x)
        if self.queue_size > self.max_batches:
            self.remove_last_batch()

    def remove_last_batch(self):
        self.q = self.q[self.batch_size :]

    def get_tensor(self):
        return torch.stack(self.q, 0)

    @property
    def queue_size(self):
        q_tensor = torch.stack(self.q, 0)
        return q_tensor.shape[0] // self.batch_size


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.n_classes = int(self.config["DATA"]["n_classes"])
        self.scale_factor = config["MODEL"]["scale_factor"]
        self.debug_plot_every = 10
        self.moco_augm = MoCoAugmentations(config)
        self.k_queue = MoCoKeyQueue(
            max_batches=config["MODEL"]["moco_queue"]["max_batches"],
            batch_size=config["DATA"]["batch_size"],
        )

    def train(self):
        self.logger.info("Started training..")
        self.initialize_encoders()

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            if epoch % self.config["OPTIM"]["save_ckpt_every"] == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (x) in pbar:
            x = x.to(self.device)
            # augment
            x_q = self.moco_augm.augment(x)
            x_k = self.moco_augm.augment(x)

            # encode
            q = self.f_q(x_q)
            k = self.f_k(x_k).detach()

            # compute similarity

            loss_dict = {}
            loss = 0
            self.write_dict_to_tb(loss_dict, self.total_iters, prefix="train")

            loss.backward()
            self.total_iters += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()

    def initialize_encoders(self):
        self.f_k = copy.deepcopy(self.model)
        self.f_q = copy.deepcopy(self.model)

    def evaluate_model(self):
        return
