import numpy as np
import os
import attridict

from visolex.utils import get_arguments, get_tokenizer
from visolex.framework_components.log import get_logger
from visolex.framework_components.student import Student
from visolex.framework_components.trainer import ViSoLexTrainer
from visolex.framework_components.data_handler import DataHandler
from visolex.framework_components.evaluator import Evaluator
from visolex.framework_components.asset_fetcher import AssetFetcher

ARGS = {
    "student_name": "visobert",
    "teacher_name": "ran",
    "training_mode": "weakly_supervised",
    "inference_model": "student",
    "metric": "f1_score",
    "num_iter": 10,
    "num_rules": 2,
    "num_epochs": 10,
    "num_unsup_epochs": 5,
    "debug": 0,
    "remove_accents": 0,
    "rm_accent_ratio": 0.0,
    "append_n_mask": 1,
    "nsw_detect": 1,
    "soft_labels": 1,
    "loss_weights": 0,
    "hard_student_rule": 1,
    "train_batch_size": 16,
    "eval_batch_size": 128,
    "unsup_batch_size": 128,
    "lowercase": 1,
    "learning_rate": 0.001,
    "fine_tuning_strategy": "flexible_lr",
    "sample_size": 8096,
    "topk": 1,
    "seed": 42,
    "downstream_task": "vihsd"
}
args = ARGS
args = attridict(args)
args.datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_data')
args.ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'model_checkpoints')
args.logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')

np.random.seed(args.seed)

suffix = f"{args.student_name}_{args.training_mode}_{args.rm_accent_ratio}"
logfile = os.path.join(args.logdir, f'log_{suffix}.log')
if os.path.exists(logfile):
    os.remove(logfile)
logger = get_logger(logfile=logfile)

ev = Evaluator(args, logger=logger)
tokenizer = get_tokenizer(args.student_name)

logger.info("Building student: {}".format(args.student_name))
normalizer = Student(args, tokenizer=tokenizer, logger=logger)

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