def detect_nsw(input_str, args=None):
    import os
    from visolex.lexnorm.detect import NswDetector

    if args is None:
        args = get_arguments()
    logger = get_logger(logfile=os.path.join(args.logdir, 'detect.log'))
    detector = NswDetector(args, logger)
    nsw_spans = detector.detect_nsw(input_str=input_str)
    return nsw_spans

def normalize_sentence(input_str, args=None, nsw_detection=True):
    import os
    from visolex.lexnorm.normalize import ViSoLexNormalizer

    if args is None:
        args = get_arguments()
    logger = get_logger(logfile=os.path.join(args.logdir, 'normalize.log'))
    normalizer = ViSoLexNormalizer(args, logger)
    self.normalizer.load()
    output = normalizer.normalize_sentence(input_str, nsw_detection)
    return output