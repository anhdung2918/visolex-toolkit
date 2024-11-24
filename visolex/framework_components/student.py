import os
from .normalizer.trainer import Trainer
from visolex.utils import sort_data
from visolex.framework_components.asset_fetcher import AssetFetcher

class Student:
    def __init__(self, args, tokenizer, logger=None):
        self.args = args
        self.logger = logger
        self.name = args.student_name
        self.tokenizer = tokenizer
        self.training_mode = args.training_mode
        self.remove_accents = args.remove_accents
        self.trainer = Trainer(args=self.args, tokenizer=self.tokenizer, logger=self.logger)

    def train(self, train_dataset, dev_dataset, mode='train'):
        assert mode in ['train', 'finetune', 'train_pseudo']
        if mode in ['train', 'finetune']:
            train_dataset = sort_data(train_dataset, remove_accents=self.remove_accents)
        dev_dataset = sort_data(dev_dataset, remove_accents=self.remove_accents)
        if mode == 'train':
            res = self.trainer.train(
                train_data=train_dataset,
                dev_data=dev_dataset,
            )
            return res
        if mode == 'finetune':
            res = self.trainer.finetune(
                train_data=train_dataset,
                dev_data=dev_dataset,
            )
            return res
        if mode == 'train_pseudo':
            res = self.trainer.train_pseudo(
                train_data=train_dataset.teacher_data if self.training_mode=='weakly_supervised' else train_dataset.student_data,
                dev_data=dev_dataset,
            )
            return res

    def predict(self, dataset, dataIter=None, inference_mode=False):
        res = self.trainer.predict(data=dataset, dataIter=dataIter, inference_mode=inference_mode)
        return res

    def inference(self, user_input):
        res = self.trainer.inference(user_input)
        return res

    def save(self, name='student'):
        # savefolder = ./model_checkpoints/visobert/weakly_supervised_0.0/student
        savefolder = os.path.join(
            self.args.ckpt_dir, self.args.student_name, f"{args.training_mode}_{args.rm_accent_ratio}", name
        )
        self.logger.info('Saving {} to {}'.format(name, savefolder))
        os.makedirs(savefolder, exist_ok=True)
        self.trainer.save(savefolder)

    def load(self, name='student', best=False):
        version = f"{name}_best" if best else f"{name}_last"
        savefolder = os.path.join(
            self.args.ckpt_dir, self.args.student_name, f"{args.training_mode}_{args.rm_accent_ratio}", version
        )
        if not os.path.exists(savefolder):
            AssetFetcher.download_model(self.args, version, self.logger)
        self.trainer.load(savefolder)