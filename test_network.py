from argparse import ArgumentParser

from pytorch_lightning import seed_everything

from PTDataSet import TorchDataset
from network import CustomNetwork


def main(hparams):
    seed_everything(11)

    test_dataset = TorchDataset(args.test_path)

    # initialize the model
    model = CustomNetwork(training=False, hparams=hparams, batch_size=hparams.batch_size,
                          test_dataset=test_dataset, num_classes=2).load_from_checkpoint("model1_0.01.ckpt")

    # get first image from dataset

    img, mask = test_dataset[1]

    output = model(img.unsqueeze(0))

    output = output.argmax(axis=1)[0]

    from batchviewer import view_batch

    view_batch(output, mask, width=1024, height=1024)


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
