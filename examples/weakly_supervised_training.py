from visolex.framework_components.trainer import ViSoLexTrainer
from visolex.framework_components.data_handler import DataHandler


ev = Evaluator(args, logger=logger)
tokenizer = get_tokenizer(args.student_name)

logger.info("Building student: {}".format(args.student_name))
student = Student(args, tokenizer=tokenizer, logger=logger)

logger.info("Loading data")
dh = DataHandler(args, tokenizer=tokenizer, logger=logger)
train_dataset = dh.load_dataset(method='train') 
dev_dataset = dh.load_dataset(method='dev')
test_dataset = dh.load_dataset(method='test')
unlabeled_dataset = dh.load_dataset(method='unlabeled')

trainer = ViSoLexTrainer(
    tokenizer, normalizer, training_args, logger,
    train_dataset, dev_dataset, test_dataset,
    unlabeled_dataset=None
)

trainer.train()