import copy
from visolex.utils import get_tokenizer, delete_special_tokens
from visolex.framework_components.student import Student

class NswDetector:
    def __init__(args, logger, tokenizer=None, normalizer=None):
        self.args = args
        self.logger = self.logger
        self.loaded = True if normalizer is None else False

        if tokenizer is None:
            self.tokenizer = get_tokenizer(self.args.student_name)
        else:
            self.tokenizer = tokenizer

        if normalizer is None:
            self.normalizer = Student(self.args, tokenizer=self.tokenizer, logger=self.logger)
            self.normalizer.load('student_best')
        else:
            self.normalizer = normalizer

    def load(self, last=False):
        if self.loaded:
            self.logger.info("Model have already been loaded")
        else:
            self.logger.info("Loading model from checkpoints")
            if last:
                self.normalizer.load('student_last')
            else:
                self.normalizer.load('student_best')
            self.loaded = True

    def concatenate_nsw_spans(nsw_spans):
        result = []
        current_span = nsw_spans[0]

        for i in range(1, len(nsw_spans)):
            next_span = nsw_spans[i]
            if current_span['end_index'] == next_span['start_index']:
                current_span['nsw'] += next_span['nsw']
                current_span['end_index'] = next_span['end_index']
            else:
                result.append(current_span)
                current_span = next_span
        result.append(current_span)

        return result

    def detect_nsw(self, input_str=None, normalizer_output=None):
        if not self.loaded:
            self.logger.info("Model have not been loaded yet!")
            self.load()

        if normalizer_output is None:
            assert (input_str is not None) & (self.normalizer is not None)
            normalizer_output = self.normalizer.inference(user_input=input_str)

        source_tokens = normalizer_output['source_tokens']
        is_nsw = normalizer_output['is_nsw']
        source_tokens, keep_indices = delete_special_tokens(source_tokens)
        is_nsw = [is_nsw[i] for i in keep_indices]
        nsw_indices = [i for i, nsw in enumerate(is_nsw) if nsw == 1]
        nsw_tokens = [source_tokens[i] for i in nsw_indices]

        nsw_spans = []
        end_index = 0
        for i in range(len(source_tokens)):
            if source_tokens[i].startswith('▁'):
                end_index += 1
            current_text = self.tokenizer.convert_tokens_to_string([source_tokens[i]])
            full_text = self.tokenizer.convert_tokens_to_string(source_tokens[:(i+1)])
            if is_nsw[i] == 1:
                if current_text:
                    nsw_spans.append({
                        'index': i,
                        'start_index': end_index,
                        'end_index': end_index + len(current_text),
                        'nsw': current_text
                    })
            end_index = len(full_text) if current_text else len(full_text) + 1

        if input_str is not None:
            copied_nsw_spans = copy.deepcopy(nsw_spans)
            concat_nsw_spans = self.concatenate_nsw_spans(copied_nsw_spans)
            return concat_nsw_spans
        return nsw_spans