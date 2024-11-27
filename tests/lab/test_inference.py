# Inference on a .csv file
import os
import attridict
from visolex.utils import get_tokenizer,bdelete_special_tokens, post_process
from visolex.framework_components.log import get_logger
from visolex.framework_components.student import Student

def decode_prediction(pred, tokenizer):
    decoded_pred = tokenizer.convert_ids_to_tokens(pred)
    pred_tokens, _ = delete_special_tokens(decoded_pred)
    pred_str = tokenizer.convert_tokens_to_string(pred_tokens)
    pred_str = post_process(pred_str)
    return pred_str

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

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sample.txt')
with open(data_path) as f:
    texts = f.readlines()

args.logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
logger = get_logger(logfile=os.path.join(args.logdir, 'test_inference.log'))

tokenizer = get_tokenizer(args.student_name)
normalizer = Student(args, tokenizer=tokenizer, logger=logger)
normalizer.load("student_best")

for text in texts:
    token_length = len(tokenizer.encode(text))
    if token_length <= 1024:
        output = normalizer.inference(user_input=text)
        pred = output['pred']
        pred_str = decode_prediction(pred, tokenizer)
    else:
        pred_str = "."

    with open (os.path.join(os.path.dirname(os.path.realpath(__file__)), 'predictions.txt'), 'w') as f:
        f.write(pred_str)
        f.write('\n')