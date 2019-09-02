# FASHION MNIST




| Classifier | Preprocessing | Fashion test accuracy | Inference time (ms/image) | 
| --- | --- | --- | --- | 
|2 Conv+pooling ~168k params | Normalization, Random cropping, Random Horizontal flip, BN | 0.9056 | 0.3 |
|5 Conv+pooling ~3M params | Normalization, BN | 0.913 | 1.1 |
|VGG16 138M params minus 5 first convolution which are frozen| Normalization| 0.9369| 4.5 |


##### Random cropping 

The semantics of the image are preserved but the activation values of the conv net are 
different. The conv net learns to associate a broader range of spatial activations with 
a certain class label and improves the robustness of the feature detectors in conv nets.

[Data Augmentation using Random Image Cropping and Patching for Deep CNNs
Ryo Takahashi, Takashi Matsubara, Kuniaki Uehara](https://arxiv.org/abs/1409.1556) 


## Args:
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

    parser.add_argument('--train-batch-size',
                        default=50,
                        help='batch size for training with Adam')

    parser.add_argument('--lr',
                        default=0.005,
                        type=float,
                        help='learning rate')

    parser.add_argument('--train-epoch',
                        default=60,
                        type=int,
                        help='number of training epoch')

    parser.add_argument('--seed',
                        default=42,
                        help='seed')

    parser.add_argument('--save-dir',
                        default='./data',
                        help='saving metrics dir')

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

### Examples:


#### Training:
`python -m eval --model-type two_conv --train-epoch 60`

#### Test:
`python -m eval --model-type two_conv --test-model-path ./data/models/two_conv.pth`
