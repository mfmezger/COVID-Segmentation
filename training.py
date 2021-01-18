from argparse import ArgumentParser

from PIL import Image
from network import CustomNetwork
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from PTDataSet import TorchDataset


def main(hparams):
    seed_everything(11)




    train_dataset = TorchDataset(args.train_path, augmentations=True)
    val_dataset = TorchDataset(args.val_path)
    test_dataset = TorchDataset(args.test_path)

    # initialize the model
    model = CustomNetwork(training=True, hparams=hparams, batch_size=hparams.batch_size,
                         train_dataset=train_dataset, val_dataset=val_dataset,
                         test_dataset=test_dataset,  num_classes=2)

    # initialize loggers
    checkpoint_callback = ModelCheckpoint(filepath='pl_model/')

    # Intialize the Trainer.
    trainer = Trainer(gpus=1, num_nodes=1,
                       profiler=True, min_epochs=1, max_epochs=hparams.max_epochs,
                      checkpoint_callback=checkpoint_callback, benchmark=True, progress_bar_refresh_rate=20)

    # start  the Training.
    trainer.fit(model)

    # activate testing
    trainer.test(model)

    trainer.save_checkpoint("model" + str(args.max_epochs) + "_" + str(args.learning_rate) + ".ckpt")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--opt', default="ranger", type=str)
    parser.add_argument('--loss', default="ce")
    parser.add_argument('--augmentations', default=False, type=bool)
    parser.add_argument('--max_epochs', default=20, type=int)
    parser.add_argument('--train_path',
                        default="/home/mfmezger/data/COVID/")
    parser.add_argument('--val_path',
                        default="/home/mfmezger/data/COVID/")
    parser.add_argument('--test_path',
                        default="/home/mfmezger/data/COVID/")
    args = parser.parse_args()

    main(args)
