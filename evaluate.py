import argparse

from moco.dataset.voc_dataset import VOCDataset
from moco.model.resnet import ResNet18Model
from moco.utils.misc import get_logger, load_config, make_artifacts_dirs
from moco.utils.trainer import Trainer


def evaluate(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = VOCDataset(cfg=config, mode="train")
    val_dataset = VOCDataset(cfg=config, mode="val")

    model = ResNet18Model(config)
    trainer.set_model(model)

    trainer.set_dataset(
        train_dataset, val_dataset, data_config=config["DATA"], shuffle_valset=True, val_set_batch_size=2
    )
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.load_checkpoint(args.ckpt)

    trainer.evaluate_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/gpt_config.yaml", help="Config path")
    parser.add_argument("ckpt", type=str)
    args = parser.parse_args()
    evaluate(args)
