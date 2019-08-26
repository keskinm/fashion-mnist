import logging


def setup_argparse_logging_level(parser):
    parser.add_argument('--logging-level',
                        default=logging.INFO,
                        type=lambda level: int(getattr(logging, level)),
                        help='Logging level')


def setup_logging(level):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=level)
