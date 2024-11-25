import copy

from .detect import NswDetector
from visolex.utils import get_tokenizer, delete_special_tokens, post_process
from visolex.framework_components.student import Student


class ViSoLexNormalizer:
    def __init__(args, logger):
        self.args = args
        self.logger = self.logger
        self.tokenizer = get_tokenizer(self.args.student_name)
        self.normalizer = Student(self.args, tokenizer=self.tokenizer, logger=self.logger)
        self.loaded = False

    def load(self, last=False):
        if self.args.inference_model == "teacher":
            print("Teacher inference is not supported yet. Load Normalizer instead")
        if last:
            self.normalizer.load("student_last")
        else:
            self.normalizer.load("student_best")
        self.loaded = True

    def normalize_sentence(self, input_str, detect_nsw=False):
        if not self.loaded:
            self.load()
        output = self.normalizer.inference(user_input=input_str)
        pred = output['pred']
        proba = output['proba']
        decoded_pred = self.tokenizer.convert_ids_to_tokens(pred)

        if detect_nsw:
            self.nsw_detector = NswDetector(self.args, self.logger, self.tokenizer, self.normalizer)
            nsw_spans = self.nsw_detector.detect_nsw(normalizer_output=output)
            nsw_indices = [span['index'] for span in nsw_spans]
            for i, nsw_idx in enumerate(nsw_indices):
                nsw_spans[i]['prediction'] = self.tokenizer.convert_tokens_to_string([decoded_pred[nsw_idx+1]])
                nsw_spans[i]['confidence_score'] = round(proba[nsw_idx+1], 4)
            pred_tokens, keep_indices = delete_special_tokens(decoded_pred)
            proba = [proba[i] for i in keep_indices]
            pred_str = self.tokenizer.convert_tokens_to_string(pred_tokens)
            pred_str = post_process(pred_str)
            copied_nsw_spans = copy.deepcopy(nsw_spans)
            concat_nsw_spans = self.nsw_detector.concatenate_nsw_spans(copied_nsw_spans)
            return nsw_spans, pred_str

        else:
            pred_str = self.tokenizer.convert_tokens_to_string(decoded_pred)
            pred_str = post_process(pred_str)
            return pred_str