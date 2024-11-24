import numpy as np
import os
import argparse

from visolex.utils import get_arguments, get_tokenizer
from visolex.framework_components.log import get_logger
from visolex.framework_components.student import Student
from visolex.framework_components.trainer import ViSoLexTrainer
from visolex.framework_components.data_handler import DataHandler
from visolex.framework_components.evaluator import Evaluator
from visolex.framework_components.asset_fetcher import AssetFetcher

def train_visolex(model_name, training_mode, rm_accent_ratio, custom_dataset=False):
    args = get_arguments()
    args.student_name = model_name
    args.training_mode = training_mode
    args.rm_accent_ratio = rm_accent_ratio
    if args.rm_accent_ratio != 0.0:
        args.remove_accents = 1

    np.random.seed(args.seed)

    # logger
    os.makedirs(args.logdir, exist_ok=True)
    suffix = f"{args.student_name}_{args.training_mode}_{args.rm_accent_ratio}"
    logfile = os.path.join(args.logdir, f'log_{suffix}.log')
    if os.path.exists(logfile):
        os.remove(logfile)
    logger = get_logger(logfile=logfile)

    ev = Evaluator(args, logger=logger)
    tokenizer = get_tokenizer(args.student_name)

    logger.info("Building student: {}".format(args.student_name))
    normalizer = Student(args, tokenizer=tokenizer, logger=logger)

    if not custom_dataset:
        logger.info("Download dataset")
        AssetFetcher.download_data(logger)

    logger.info("Loading data")
    dh = DataHandler(args, tokenizer=tokenizer, logger=logger)
    train_dataset = dh.load_dataset(method='train') 
    dev_dataset = dh.load_dataset(method='dev')
    test_dataset = dh.load_dataset(method='test')
    unlabeled_dataset = dh.load_dataset(method='unlabeled')

    trainer = ViSoLexTrainer(
        args, dh, tokenizer, normalizer, logger, ev,
        train_dataset, dev_dataset, test_dataset,
        unlabeled_dataset=unlabeled_dataset
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Normalizer short name", type=str, default='bartpho')
    parser.add_argument("--training_mode", help="Training mode ['supervised', 'semi_supervised', 'weakly_supervised']", type=str, default='weakly_supervised')
    parser.add_argument("--rm_accent_ratio", default=0.0, type=float, help="The ratio of character in a sentence to remove accents")
    parser.add_argument("--custom_data", action="store_true", help="Whether to train model on custom data")
    args = parser.parse_args()
    
    train_visolex(args.model_name, args.training_mode, args.rm_accent_ratio, args.custom_data)