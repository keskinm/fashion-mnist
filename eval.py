import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models.vgg import VGG, vgg16
from models.two_layers import TwoLayers
import logging
from utils.logging import setup_logging, setup_argparse_logging_level
import argparse
import time
import os

logger = logging.getLogger(__name__)


class FMModelsEvaluator:
    def __init__(self, train_epoch, lr, train_batch_size, test, model_type, threshold_validation_accuracy, seed, save_dir):
        self.train_epoch = train_epoch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr[0]
        self.train_batch_size = train_batch_size
        self.test = test
        self.model_type = model_type
        self.threshold_validation_accuracy = threshold_validation_accuracy
        if seed is not None:
            torch.manual_seed(seed)
        self.save_dir = save_dir
        self.loss_plots_dir = os.path.join(save_dir, 'losses_plots')
        self.save_model_dir_path = os.path.join(save_dir, 'models')
        os.makedirs(self.loss_plots_dir, exist_ok=True)
        os.makedirs(self.save_model_dir_path, exist_ok=True)

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor()])

        train_val_set = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(train_val_set, (50000, 10000))
        test_set = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

        train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
        val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=True)
        test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=True)

        return train_set_loader, val_set_loader, test_set_loader

    def eval(self):
        train_set_loader, val_set_loader, test_set_loader = self.prepare_data()
        # if self.model_type == 'vgg':
        #     models_subtypes = [vgg16(pretrained=True, num_classes=1000).to(self.device),
        #                        vgg16(num_classes=10).to(self.device)]
        #
        # elif self.model_type == 'two_layers':
        #     models_subtypes = [TwoLayers(num_classes=10).to(self.device)]
        #
        # else:
        #     models_subtypes = None

        models = {'two_layers': TwoLayers(num_classes=10).to(self.device),
                  'vgg16_pretrained': vgg16(pretrained=True, num_classes=1000).to(self.device),
                  'vgg16': vgg16(num_classes=10).to(self.device)}

        for model_name, model in models.items():
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0006, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            self.train(model, train_set_loader, optimizer=optimizer, criterion=criterion, model_name=model_name)
            accuracy = self.compute_accuracy(model, val_set_loader)
            self.dump_metrics(accuracy, model_name)

    def run(self):
        self.eval()
        # self.load_model(None, None, './data/models/two_layers.pth')
        # self.test(model, test_set_loader)

    def train(self, model, train_set_loader, optimizer, criterion, model_name):
        loss = None
        epoch = None
        losses = []

        # self.freeze_params(model)

        for epoch in range(1, self.train_epoch + 1):
            for batch_id, (image, label) in enumerate(train_set_loader):
                print(batch_id)
                label, image = label.to(self.device), image.to(self.device)
                output = model(image)
                loss = criterion(output, label)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_id % 1000 == 0:
                    print('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, self.train_epoch))

                # if batch_id == 2:
                #     break

        self.plot_losses(losses, model_name)
        self.save_model(epoch, loss, model, optimizer, model_name)

        # return model

    def save_model(self, epoch, loss, model, optimizer, model_name):
        # save_model_file_path = os.path.join(self.save_model_dir_path, '{}.pth'.format(model_name))
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss}, save_model_file_path)
        save_model_file_path = os.path.join(self.save_model_dir_path, '{}.pth'.format(model_name))
        torch.save(model, save_model_file_path)

    def plot_losses(self, losses, model_name):
        plt.plot(range(len(losses)), losses)
        plot_file_path = os.path.join(self.loss_plots_dir, model_name)
        plt.savefig(plot_file_path)

    def freeze_params(self, model):
        # for name, param in model.named_parameters():
        #     if name in ['features.0.weight', 'features.0.bias']:
        #         param.requires_grad = False
        pass

    def compute_accuracy(self, model, data_set):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for image, label in data_set:
                image = image.to(self.device)
                label = label.to(self.device)
                outputs = model(image)
                predicted = torch.argmax(outputs, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            accuracy = 100 * correct / total
            model.train()
            return accuracy

    def dump_metrics(self, accuracy, model_name):
        metrics_file_path = os.path.join(self.save_dir, 'metrics.txt')
        with open(metrics_file_path, "a") as opened_metrics_file:
            opened_metrics_file.write("model name:{} \naccuracy: {}\n\n".format(model_name, accuracy))

    def load_model(self, model, optimizer, model_params_path):
        # model = model()
        # optimizer = optimizer()
        #
        # checkpoint = torch.load(model_params_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        model = torch.load(model_params_path)
        state_dict = model.state_dict()


def main():
    parser = argparse.ArgumentParser(prog='Fashion MNIST Models Evaluator')
    setup_argparse_logging_level(parser)

    parser.add_argument(
        '--model-type',
        choices=['vgg, two_layers, sixlayers'],
        default='two_layers',
        help=
        ''
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help=
        ''
    )

    parser.add_argument(
        '--train-batch-size',
        default=[50],
        help=
        ''
    )

    parser.add_argument(
        '--lr',
        default=[0.005],
        help=
        ''
    )

    parser.add_argument(
        '--train-epoch',
        default=5,
        type=int,
        help=
        ''
    )

    parser.add_argument(
        '--threshold-validation-accuracy',
        default=0.80,
        help=
        ''
    )

    parser.add_argument(
        '--seed',
        default=42,
        help=
        ''
    )

    parser.add_argument(
        '--save-dir',
        default='./data',
        help=
        ''
    )

    args = parser.parse_args()
    args = vars(args)
    setup_logging(args.pop('logging_level'))
    evaluator = FMModelsEvaluator(**args)
    evaluator.run()


if __name__ == "__main__":
    main()
