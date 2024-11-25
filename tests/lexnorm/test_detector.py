import unittest
from unittest import TestCase
import attridict
from visolex import NswDetector
from visolex.framework_components.log import get_logger

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

class TestNSWDetector(TestCase):
    def test_1(self):
        # Implicitly load model
        args = ARGS
        args = attridict(args)
        args.ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'model_checkpoints'
        )
        args.logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        os.makedirs(args.logdir, exist_ok=True)

        logger = get_logger(
            logfile=os.path.join(args.logdir, 'test_detector_1.log')
        )
        nsw_detector = NswDetector(args, logger)
        input_str = "sao lỗi j mà khó chệu dzô cùng"
        nsw_spans = nsw_detector.detect_nsw(input_str)
        logger.info("NSW DETECTION")
        for i in range(len(nsw_spans)):
            logger.info(f"{i + 1}. NSW '{nsw_spans[i]['nsw']}' start from index {nsw_spans[i]['start_index']} to index {nsw_spans[i]['end_index']}")
        expected_nsw_spans = [
            {'index': 0, 'start_index': 8, 'end_index': 9, 'nsw': 'j'},
            {'index': 1, 'start_index': 17, 'end_index': 21, 'nsw': 'chệu'},
            {'index': 2, 'start_index': 22, 'end_index': 25, 'nsw': 'dzô'}
        ]
        self.assertDictEqual(nsw_spans[0], expected_nsw_spans[0])
        self.assertDictEqual(nsw_spans[1], expected_nsw_spans[1])
        self.assertDictEqual(nsw_spans[2], expected_nsw_spans[2])

    def test_2(self):
        # Explicitly load model
        args = ARGS
        args = attridict(args)
        args.ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'model_checkpoints'
        )
        args.logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
        os.makedirs(args.logdir, exist_ok=True)

        logger = get_logger(
            logfile=os.path.join(args.logdir, 'test_detector_2.log')
        )
        nsw_detector = NswDetector(args, logger)
        nsw_detector.load()
        input_str = "trong phần kết luận nếu có thể thì em nen tóm luoc lại các kết luận có dinh lượng hơn thay vì chung chung"
        nsw_spans = nsw_detector.detect_nsw(input_str)
        logger.info("NSW DETECTION")
        for i in range(len(nsw_spans)):
            logger.info(f"{i + 1}. NSW '{nsw_spans[i]['nsw']}' start from index {nsw_spans[i]['start_index']} to index {nsw_spans[i]['end_index']}")
        expected_nsw_spans = [
            {'index': 0, 'start_index': 38, 'end_index': 41, 'nsw': 'nen'},
            {'index': 1, 'start_index': 46, 'end_index': 50, 'nsw': 'luoc'},
            {'index': 2, 'start_index': 71, 'end_index': 75, 'nsw': 'dinh'}
        ]
        self.assertDictEqual(nsw_spans[0], expected_nsw_spans[0])
        self.assertDictEqual(nsw_spans[1], expected_nsw_spans[1])
        self.assertDictEqual(nsw_spans[2], expected_nsw_spans[2])

if __name__ == '__main__':
    unittest.main()