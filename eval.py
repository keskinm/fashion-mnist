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
    def __init__(self, train_epoch, lr, train_batch_size, test_model,
                 model_type, seed, save_dir, resume_model, optimizer,
                 dump_metrics_frequency, threshold_validation_accuracy):
        self.train_epoch = train_epoch
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.model_type = model_type
        if seed is not None:
            torch.manual_seed(seed)
        self.save_dir = save_dir
        self.loss_plots_dir = os.path.join(save_dir, 'losses_plots')
        self.save_model_dir_path = os.path.join(save_dir, 'models')
        os.makedirs(self.loss_plots_dir, exist_ok=True)
        os.makedirs(self.save_model_dir_path, exist_ok=True)

        self.models = {
            'two_layers':
            TwoLayers(num_classes=10).to(self.device),
            'vgg16_pretrained':
            vgg16(pretrained=True, num_classes=10).to(self.device),
            'vgg16':
            vgg16(num_classes=10).to(self.device)
        }

        if test_model is not None:
            self.test_model_name, self.model_to_test_path = test_model
            self.test_model = self.models[self.test_model_name]
        else:
            self.model_to_test_path = None

        if resume_model is not None:
            self.resume_model_name, self.model_to_resume_path = resume_model
            self.resume_model = self.models[self.resume_model_name]
        else:
            self.model_to_resume_path = None

        self.optimizer = optimizer
        self.dump_metrics_frequency = dump_metrics_frequency
        self.threshold_validation_accuracy = threshold_validation_accuracy

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor()])

        train_val_set = datasets.FashionMNIST('./data',
                                              download=True,
                                              train=True,
                                              transform=transform)
        train_set, val_set = torch.utils.data.random_split(
            train_val_set, (50000, 10000))
        test_set = datasets.FashionMNIST('./data',
                                         download=True,
                                         train=False,
                                         transform=transform)

        train_set_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.train_batch_size, shuffle=True)
        val_set_loader = torch.utils.data.DataLoader(val_set,
                                                     batch_size=50,
                                                     shuffle=True)
        test_set_loader = torch.utils.data.DataLoader(test_set,
                                                      batch_size=50,
                                                      shuffle=True)

        return train_set_loader, val_set_loader, test_set_loader

    def eval(self, train_set_loader, val_set_loader):
        if self.model_type:
            models = {
                model_name: model
                for model_name, model in self.models.items()
                if self.model_type in model_name
            }

        else:
            models = self.models

        for model_name, model in models.items():
            criterion, optimizer = self.init_optimizer(model, model_name)
            self.train(model=model,
                       train_set_loader=train_set_loader,
                       val_set_loader=val_set_loader,
                       optimizer=optimizer,
                       criterion=criterion,
                       model_name=model_name)

    def init_optimizer(self, model, model_name):
        if isinstance(model, VGG) and 'pretrained' in model_name:
            self.freeze_params(model)

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                         lr=self.lr)

        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                        lr=self.lr,
                                        momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        return criterion, optimizer

    def run(self):
        train_set_loader, val_set_loader, test_set_loader = self.prepare_data()

        if self.model_to_test_path is not None:
            model = self.load_model(model=self.test_model,
                                    optimizer=None,
                                    model_params_path=self.model_to_test_path)

            inference_time_start = time.time()
            self.compute_accuracy(model, test_set_loader)
            inference_time = time.time() - inference_time_start
            logging.info(inference_time)

        elif self.model_to_resume_path is not None:
            model = self.resume_model
            criterion, optimizer = self.init_optimizer(model,
                                                       self.resume_model_name)
            model, optimizer, loss, epoch = self.load_model(
                model=model,
                optimizer=optimizer,
                model_params_path=self.model_to_resume_path)
            self.train(model=model,
                       train_set_loader=train_set_loader,
                       val_set_loader=val_set_loader,
                       optimizer=optimizer,
                       criterion=criterion,
                       model_name=self.resume_model_name,
                       loss=loss,
                       epoch_start_idx=epoch)

        else:
            self.eval(train_set_loader, val_set_loader)

    def train(self,
              model,
              train_set_loader,
              val_set_loader,
              optimizer,
              criterion,
              model_name,
              loss=None,
              epoch_start_idx=1):

        epoch_n = self.train_epoch
        model = model.train()
        epoch = None
        losses = []

        for epoch in range(epoch_start_idx, epoch_n + 1):
            for batch_id, (image, label) in enumerate(train_set_loader):
                logger.info(batch_id)
                label, image = label.to(self.device), image.to(self.device)
                output = model(image)
                loss = criterion(output, label)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_id != 0) and (batch_id % self.dump_metrics_frequency == 0):
                    accuracy = self.compute_accuracy(model, val_set_loader)
                    self.dump_metrics_and_save_model(accuracy, epoch, epoch_n,
                                                     loss, losses, model,
                                                     model_name, optimizer,
                                                     batch_id)
                    if accuracy >= self.threshold_validation_accuracy:
                        break
            else:
                continue

            break

    def dump_metrics_and_save_model(self, accuracy, epoch, epoch_n, loss,
                                    losses, model, model_name, optimizer,
                                    batch_id):
        self.dump_accuracy(accuracy, model_name, epoch, batch_id)
        self.save_model(epoch, loss, model, optimizer, model_name)
        self.plot_losses(losses, model_name)
        logger.info('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch,
                                                       epoch_n))

    def save_model(self, epoch, loss, model, optimizer, model_name):
        save_model_file_path = os.path.join(self.save_model_dir_path,
                                            '{}.pth'.format(model_name))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_model_file_path)

    def plot_losses(self, losses, model_name):
        plt.plot(range(len(losses)), losses)
        plot_file_path = os.path.join(self.loss_plots_dir, model_name)
        plt.savefig(plot_file_path)

    @staticmethod
    def freeze_params(model):
        for name, param in model.named_parameters():
            low_layers = ['0.', '2.', '5.', '7.', '10.']
            low_layer_in_name = False
            for layer_idx in low_layers:
                if layer_idx in name:
                    low_layer_in_name = True
            if low_layer_in_name and ('features' in name):
                param.requires_grad = False

    def compute_accuracy(self, model, data_set):
        model = model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_id, (image, label) in enumerate(data_set):
                logger.info(batch_id)
                image = image.to(self.device)
                label = label.to(self.device)
                outputs = model(image)
                predicted = torch.argmax(outputs, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            accuracy = 100 * correct / total
        logger.info(accuracy)
        return accuracy

    def dump_accuracy(self, accuracy, model_name, epoch, batch_idx):
        metrics_dir_path = os.path.join(self.save_dir, 'metrics')
        metrics_file_path = os.path.join(metrics_dir_path,
                                         '{}.txt'.format(model_name))
        with open(metrics_file_path, "a") as opened_metrics_file:
            opened_metrics_file.write(
                "epoch:{epoch} batch_idx:{batch_idx} val_accuracy:{val_accuracy}\n"
                .format(epoch=epoch,
                        batch_idx=batch_idx,
                        val_accuracy=accuracy))

    def load_model(self, model, optimizer, model_params_path):
        checkpoint = torch.load(model_params_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if self.model_to_resume_path is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            return model.to(self.device), optimizer, loss, epoch

        else:
            return model.to(self.device)


def main():
    parser = argparse.ArgumentParser(prog='Fashion MNIST Models Evaluator')
    setup_argparse_logging_level(parser)

    parser.add_argument('--model-type',
                        choices=['vgg', 'two_layers', 'sixlayers'],
                        default='two_layers',
                        help='')

    parser.add_argument('-t',
                        '--test-model',
                        nargs='+',
                        help='model path and model name',
                        default=None)

    parser.add_argument('-r',
                        '--resume-model',
                        nargs='+',
                        help='model path and model name',
                        default=None)

    parser.add_argument('--train-batch-size', default=50, help='')

    parser.add_argument('--lr', default=0.005, type=float, help='')

    parser.add_argument('--train-epoch', default=5, type=int, help='')

    parser.add_argument('--seed', default=42, help='')

    parser.add_argument('--save-dir', default='./data', help='')

    parser.add_argument('--optimizer',
                        choices=['adam', 'sgd'],
                        default='adam',
                        help='')

    parser.add_argument('--dump-metrics-frequency',
                        metavar='Batch_n',
                        default='200',
                        type=int,
                        help='dump metrics every Batch_n batches')

    parser.add_argument('--threshold-validation-accuracy',
                        default='0.95',
                        type=float,
                        help='dump metrics every Batch_n batches')

    args = parser.parse_args()
    args = vars(args)
    setup_logging(args.pop('logging_level'))
    evaluator = FMModelsEvaluator(**args)
    evaluator.run()


if __name__ == "__main__":
    main()
