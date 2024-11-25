import os
import unittest
from unittest import TestCase
from visolex.framework_components.asset_fetcher import AssetFetcher
from visolex.framework_components.log import get_logger
from visolex.utils import get_arguments

class TestAssetFectcher(TestCase):
    def test_data_fetcher(self):
        logfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs/test_data_fetcher.log')
        logger = get_logger(logfile=logfile)
        AssetFetcher.download_data(
            logger, 
            saved_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_dataset')
        )

    def test_model_fetcher(self):
        logfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs/test_model_fetcher.log')
        logger = get_logger(logfile=logfile)
        args = get_arguments()
        AssetFetcher.download_model(
            args=args, version='student_best', logger=logger, 
            ckpt_dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )

    def test_remove_data(self):
        logfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs/test_remove_data.log')
        logger = get_logger(logfile=logfile)
        args = get_arguments()
        AssetFetcher.remove(
            args, logger, 
            saved_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_dataset'),
            asset='dataset'
        )

    def test_remove_model(self):
        logfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs/test_remove_model.log')
        logger = get_logger(logfile=logfile)
        args = get_arguments()
        AssetFetcher.remove(
            args, logger, 
            saved_dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )

if __name__ == '__main__':
    unittest.main()