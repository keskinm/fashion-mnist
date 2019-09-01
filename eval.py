import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models.vgg import vgg, VGG
from models.two_conv_model import two_conv
from models.five_conv_model import five_conv
import logging
from utils.logging import setup_logging, setup_argparse_logging_level
from utils.dataset import DatasetTransformer, CenterReduce, compute_mean_std
import argparse
import time
import os

logger = logging.getLogger(__name__)


class FMModelsEvaluator:
    def __init__(self, train_epoch, lr, train_batch_size, test_model_path,
                 model_type, model_cfg, seed, save_dir, resume_model_path,
                 optimizer, dump_metrics_frequency,
                 threshold_validation_accuracy, num_threads, standardize,
                 scale, augment):
        self.train_epoch = train_epoch
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.model_type = model_type if model_type is not None else None
        if seed is not None:
            torch.manual_seed(seed)
        self.save_dir = save_dir
        self.loss_plots_dir = os.path.join(save_dir, 'losses_plots')
        self.save_model_dir_path = os.path.join(save_dir, 'models')
        self.metrics_dir_path = os.path.join(self.save_dir, 'metrics')

        os.makedirs(self.loss_plots_dir, exist_ok=True)
        os.makedirs(self.save_model_dir_path, exist_ok=True)
        os.makedirs(self.metrics_dir_path, exist_ok=True)

        self.model_to_test_path = test_model_path
        self.model_to_resume_path = resume_model_path

        self.optimizer = optimizer
        self.dump_metrics_frequency = dump_metrics_frequency
        self.threshold_validation_accuracy = threshold_validation_accuracy
        self.num_threads = num_threads
        self.standardize = standardize
        self.scale = scale

        self.train_set_loader, self.augmented_train_set_loader, self.val_set_loader, self.test_set_loader = self.prepare_data(
        )

        self.model_cfg = model_cfg
        self.augment = augment

    def prepare_data(self):
        pil_train_val_set = datasets.FashionMNIST('./data',
                                                  download=True,
                                                  train=True,
                                                  transform=None)
        pil_train_set, pil_val_set = torch.utils.data.random_split(
            pil_train_val_set, (50000, 10000))
        pil_test_set = datasets.FashionMNIST('./data',
                                             download=True,
                                             train=False,
                                             transform=None)

        if self.standardize:
            standardazing_dataset = DatasetTransformer(pil_train_val_set,
                                                       transforms.ToTensor(),
                                                       self.scale)
            standardizing_loader = torch.utils.data.DataLoader(
                dataset=standardazing_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_threads)

            # Compute mean and variance from the training set
            mean_train_tensor, std_train_tensor = compute_mean_std(
                standardizing_loader)

            normalization_function = CenterReduce(mean_train_tensor,
                                                  std_train_tensor)

            to_tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: normalization_function(x))
            ])
            augment_to_tensor_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: normalization_function(x))
            ])

        else:
            to_tensor_transform = transforms.Compose([transforms.ToTensor()])

            augment_to_tensor_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        train_set = DatasetTransformer(pil_train_set, to_tensor_transform,
                                       self.scale)
        augmented_train_set = DatasetTransformer(pil_train_set,
                                                 augment_to_tensor_transform,
                                                 self.scale,
                                                 augment_prob=0.5)

        valid_set = DatasetTransformer(pil_val_set, to_tensor_transform,
                                       self.scale)
        test_set = DatasetTransformer(pil_test_set, to_tensor_transform,
                                      self.scale)

        train_set_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_threads)

        augmented_train_set_loader = torch.utils.data.DataLoader(
            augmented_train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_threads)

        val_set_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=50,
            shuffle=True,
            num_workers=self.num_threads)

        test_set_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=50,
            shuffle=False,
            num_workers=self.num_threads)

        return train_set_loader, augmented_train_set_loader, val_set_loader, test_set_loader

    def define_model(self):
        if self.model_type == 'vgg':
            if self.model_cfg:
                model = vgg(**self.model_cfg,
                            cfg='A',
                            progress=True,
                            num_classes=10)
            else:
                model = vgg(cfg='A',
                            batch_norm=False,
                            pretrained=True,
                            progress=True,
                            num_classes=10)

        elif self.model_type == 'five_conv':
            self.model_cfg.pop('pretrained')
            if self.model_cfg:
                model = five_conv(**self.model_cfg, cfg='A')
            else:
                model = five_conv(cfg='A', batch_norm=False)

        else:
            self.model_cfg.pop('pretrained')
            if self.model_cfg:
                model = two_conv(**self.model_cfg, cfg='A')
            else:
                model = two_conv(cfg='A', batch_norm=False)

        if self.augment:
            model.model_name += '_augment'
        if self.scale:
            model.model_name += '_scale'
        if self.standardize:
            model.model_name += '_standardize'

        return model.to(self.device)

    def init_optimizer(self, model):
        if isinstance(model, VGG):
            if model.pretrained:
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
        model = self.define_model()

        if self.model_to_test_path is not None:
            model = self.load_model(model=model,
                                    optimizer=None,
                                    model_params_path=self.model_to_test_path)

            inference_time_start = time.time()
            self.compute_accuracy(model, self.test_set_loader)
            inference_time = time.time() - inference_time_start
            logging.info(inference_time)

        else:
            train_set_loader = self.augmented_train_set_loader if self.augment else self.train_set_loader

            if self.model_to_resume_path is not None:
                criterion, optimizer = self.init_optimizer(model)

                model, optimizer, losses, epoch = self.load_model(
                    model=model,
                    optimizer=optimizer,
                    model_params_path=self.model_to_resume_path)

                self.train(model=model,
                           train_set_loader=train_set_loader,
                           val_set_loader=self.val_set_loader,
                           optimizer=optimizer,
                           criterion=criterion,
                           losses=losses,
                           epoch_start_idx=epoch)
            else:
                train_set_loader = self.augmented_train_set_loader if self.augment else self.train_set_loader

                criterion, optimizer = self.init_optimizer(model)
                self.train(model=model,
                           train_set_loader=train_set_loader,
                           val_set_loader=self.val_set_loader,
                           optimizer=optimizer,
                           criterion=criterion)

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1**(epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self,
              model,
              train_set_loader,
              val_set_loader,
              optimizer,
              criterion,
              losses=None,
              epoch_start_idx=0):

        batch_id = None
        loss = None

        if losses is None:
            losses = []

        epoch_n = self.train_epoch
        model = model.train()

        for epoch in range(epoch_start_idx + 1, epoch_n + 1):
            self.adjust_learning_rate(optimizer, epoch)
            for batch_id, (image, label) in enumerate(train_set_loader):
                logger.info(batch_id)
                label, image = label.to(self.device), image.to(self.device)
                output = model(image)
                loss = criterion(output, label)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_id != 0) and (batch_id %
                                        self.dump_metrics_frequency == 0):
                    accuracy = self.compute_accuracy(model, val_set_loader)
                    self.dump_metrics_and_save_model(accuracy, epoch, epoch_n,
                                                     loss, losses, model,
                                                     optimizer, batch_id)

                    if accuracy >= self.threshold_validation_accuracy:
                        break
            else:
                accuracy = self.compute_accuracy(model, val_set_loader)
                self.dump_metrics_and_save_model(accuracy, epoch, epoch_n,
                                                 loss, losses, model,
                                                 optimizer, batch_id)
                continue

            break

    def dump_metrics_and_save_model(self, accuracy, epoch, epoch_n, loss,
                                    losses, model, optimizer, batch_id):
        self.dump_accuracy(accuracy, epoch, batch_id, model.model_name)
        self.save_model(epoch, losses, model, optimizer)
        self.plot_losses(losses, model.model_name)
        logger.info('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch,
                                                       epoch_n))

    def save_model(self, epoch, losses, model, optimizer):
        save_model_file_path = os.path.join(self.save_model_dir_path,
                                            '{}.pth'.format(model.model_name))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
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
            accuracy = correct / total
        logger.info(accuracy)
        return accuracy

    def dump_accuracy(self, accuracy, epoch, batch_idx, model_name):
        metrics_file_path = os.path.join(self.metrics_dir_path,
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
            losses = checkpoint['losses']
            return model.to(self.device), optimizer, losses, epoch

        else:
            return model.to(self.device)


def main():
    parser = argparse.ArgumentParser(prog='Fashion MNIST Models Evaluator')
    setup_argparse_logging_level(parser)

    parser.add_argument('--model-type',
                        choices=['vgg', 'two_conv', 'five_conv'],
                        required=True,
                        help='')

    parser.add_argument('-t',
                        '--test-model-path',
                        default=None,
                        type=str,
                        help='model path')

    parser.add_argument('-r',
                        '--resume-model-path',
                        default=None,
                        type=str,
                        help='model path')

    parser.add_argument('--train-batch-size', default=50, help='batch size for training with Adam')

    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')

    parser.add_argument('--train-epoch', default=60, type=int, help='number of training epoch')

    parser.add_argument('--seed', default=42, help='seed')

    parser.add_argument('--save-dir', default='./data', help='saving metrics dir')

    parser.add_argument('--optimizer',
                        choices=['adam', 'sgd'],
                        default='adam',
                        help='')

    parser.add_argument('--dump-metrics-frequency',
                        metavar='Batch_n',
                        default='600',
                        type=int,
                        help='Dump metrics every Batch_n batches')

    parser.add_argument(
        '--threshold-validation-accuracy',
        default='0.95',
        type=float,
        help='Threshold validation to reach for stopping training')

    parser.add_argument(
        '--num-threads',
        default='0',
        type=int,
        help='Number of CPU to use for processing mini batches')

    parser.add_argument('--scale',
                        action='store_true',
                        help='scale input in [0-1] range')

    parser.add_argument(
        '--standardize',
        action='store_true',
        help='Subtract each instance by mean of data and divide by std')

    parser.add_argument('--augment',
                        action='store_true',
                        help='Use data augmentation')

    parser.add_argument('--pretrained',
                        action='store_true',
                        help='Use pretrained weights for VGG')

    parser.add_argument('--batch-norm',
                        action='store_true',
                        help='Use batch norm')

    args = parser.parse_args()
    args = vars(args)

    model_cfg_keys = ('pretrained', 'batch_norm')
    model_cfg = {k: args[k] for k in model_cfg_keys if k in args}
    args['model_cfg'] = model_cfg
    for key in model_cfg_keys:
        args.pop(key)

    setup_logging(args.pop('logging_level'))
    evaluator = FMModelsEvaluator(**args)
    evaluator.run()


if __name__ == "__main__":
    main()
