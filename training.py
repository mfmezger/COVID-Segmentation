from argparse import ArgumentParser

from network import CustomNetwork
from PTDataSet import TorchDataset

from PIL import Image
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


def main(hparams):
    seed_everything(11)

    # 2) --> from PTDataSet import TorchDataset
    train_dataset = TorchDataset(args.train_path, augmentations=True) # legt Objekt train_dataset an (ruft konstruktor von TorchDataset auf )
    val_dataset = TorchDataset(args.val_path) 
    test_dataset = TorchDataset(args.test_path) 

    # 6) initialize the model --> from network import CustomNetwork
    model = CustomNetwork(training=True, hparams=hparams, batch_size=hparams.batch_size, # legt ein Objekt in CustomNetwork an
                         train_dataset=train_dataset, val_dataset=val_dataset, # wegen **kwargs müssen Namen zu den Vaiablen angegeben werden
                         test_dataset=test_dataset,  num_classes=2) # übergibt die kompletten Objekte train_dataset, val_dataset, test_dataset an CostumNetwork

    # initialize loggers ??????
    checkpoint_callback = ModelCheckpoint(filepath='pl_model/')

    # Intialize the Trainer.
    trainer = Trainer(gpus=1, num_nodes=1,
                       profiler=True, min_epochs=1, max_epochs=hparams.max_epochs,
                      checkpoint_callback=checkpoint_callback, benchmark=True, progress_bar_refresh_rate=20)

    # start  the Training.
    trainer.fit(model)

    # activate testing
    trainer.test(model)

    # speichert Model
    trainer.save_checkpoint("model" + str(args.max_epochs) + "_" + str(args.learning_rate) + ".ckpt")


if __name__ == '__main__': # 1) 
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--opt', default="ranger", type=str)
    parser.add_argument('--loss', default="ce")
    parser.add_argument('--augmentations', default=False, type=bool)
    parser.add_argument('--max_epochs', default=2, type=int)
    parser.add_argument('--train_path',
                        default="/home/wolfda/COVID-19-20_v2/Train_tensor_slices_filter")
    parser.add_argument('--val_path',
                        default="/home/wolfda/COVID-19-20_v2/Train_tensor_slices_filter")
    parser.add_argument('--test_path',
                        default="/home/wolfda/COVID-19-20_v2/Train_tensor_slices_filter")
    args = parser.parse_args()

    main(args)
