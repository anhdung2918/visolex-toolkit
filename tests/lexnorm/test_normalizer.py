import unittest
from unittest import TestCase
import attridict
from visolex import ViSoLexNormalizer

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

class TestViSoLexNormalizer(TestCase):
    def test_1(self):
        # Normalize sentence without NSW Detection
        args = ARGS
        args = attridict(args)
        args.ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'model_checkpoints'
        )
        args.logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        os.makedirs(args.logdir, exist_ok=True)
        logger = get_logger(logfile=os.path.join(args.logdir, 'test_normalizer_1.log'))

        normalizer = ViSoLexNormalizer(args, logger)
        normalizer.load()
        input_str = "sao lỗi j mà khó chệu dzô cùng"
        pred_str = normalizer.normalize_sentence(input_str)
        normalized_str = "Sao lỗi gì mà khó chịu vô cùng."
        logger.info(f"NORMALIZED SENTENCE: {normalized_str}")
        self.assertEqual(pred_str, normalized_str)
    
    def test_2(self):
        # Normalize sentence with NSW Detection
        args = ARGS
        args = attridict(args)
        args.ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'model_checkpoints'
        )
        args.logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        os.makedirs(args.logdir, exist_ok=True)
        logger = get_logger(logfile=os.path.join(args.logdir, 'test_normalizer_1.log'))
        
        normalizer = ViSoLexNormalizer(args, logger)
        normalizer.load()
        input_str = "sao lỗi j mà khó chệu dzô cùng"
        nsw_spans, pred_str = normalizer.normalize_sentence(input_str, nsw_detect=True)
        normalized_str = "Sao lỗi gì mà khó chịu vô cùng."
        expected_nsw_spans = [
            {'index': 0, 'nsw': 'j', 'prediction': 'gì'},
            {'index': 1, 'nsw': 'ch', 'prediction': 'chịu'},
            {'index': 2, 'nsw': 'ệ', 'prediction': '<space>'},
            {'index': 3, 'nsw': 'u', 'prediction': '<space>'},
            {'index': 4, 'nsw': 'dz', 'prediction': 'vô'},
            {'index': 5, 'nsw': 'ô', 'prediction': '<space>'}
        ]
        self.assertEqual(pred_str, normalized_str)
        for i in range(len(nsw_spans)):
            self.assertDictEqual(nsw_spans[i]['nsw'], expected_nsw_spans[i]['nsw'])
            self.assertDictEqual(nsw_spans[i]['prediction'], expected_nsw_spans[i]['prediction'])

if __name__ == '__main__':
    unittest.main()