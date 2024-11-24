import os
import shutil
import tempfile
from tqdm import tqdm as _tqdm
import requests
from urllib.parse import urlparse
import zipfile
from pathlib import Path
from visolex.global_variables import PRETRAINED_TOKENIZER_MAP, GIT_DOWNLOAD_URL, CKPT_DIR, DATASET_DIR

class Tqdm:
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        """
        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {
            'mininterval': Tqdm.default_mininterval,
            **kwargs
        }

        return _tqdm(*args, **new_kwargs)

def cached_path(url_or_filename, saved_zipfile_path, logger):
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, saved_zipfile_path, logger)
    elif parsed.scheme == '' and os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

def get_from_cache(url, saved_zipfile_path, logger):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    parent_dir = os.path.dirname(saved_zipfile_path)
    os.makedirs(parent_dir, exist_ok=True)
    # make HEAD request to check ETag
    response = requests.head(url)

    # (anhv: 27/12/2020) github release assets return 302
    if response.status_code not in [200, 302]:
        raise IOError("HEAD request failed for url {}".format(url))

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not os.path.exists(saved_zipfile_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        fd, temp_filename = tempfile.mkstemp()
        logger.info(f"{url} not found in cache, downloading to {temp_filename}")

        # GET file object
        req = requests.get(url, stream=True)
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        with open(temp_filename, 'wb') as temp_file:
            for chunk in req.iter_content(chunk_size=1024*1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)

        progress.close()

        logger.info(f"copying {temp_filename} to cache at {saved_zipfile_path}")
        shutil.copyfile(temp_filename, saved_zipfile_path)
        logger.info(f"removing temp file {temp_filename}")
        os.close(fd)
        os.remove(temp_filename)

    return saved_zipfile_path

class AssetFetcher:

    @staticmethod
    def download_data(logger):
        if os.path.exists(DATASET_DIR):
            logger.info(f"Dataset is already existed in {DATASET_DIR}")
            return

        url = f"{GIT_DOWNLOAD_URL}/dataset.zip"
        saved_zipfile_path = DATASET_DIR + '.zip'
        cached_path(url, saved_zipfile_path, logger)
        
        with zipfile.ZipFile(saved_zipfile_path) as zip_file:
            zip_file.extractall(os.path.dirname(DATASET_DIR))
        os.remove(Path(saved_zipfile_path))
        return

    @staticmethod
    def download_model(args, version, logger):
        zipfile_name = f'{args.student_name}_{args.training_mode}_{args.rm_accent_ratio}_{version}'
        if args.student_name not in PRETRAINED_TOKENIZER_MAP.keys():
            logger.info(f"No matching distribution found for '{args.student_name}'")
            return
        
        saved_dir = os.path.join(
            CKPT_DIR, args.student_name, f"{args.training_mode}_{args.rm_accent_ratio}"
        )
        saved_model_path = os.path.join(saved_dir, version)
        if os.path.exists(saved_model_path):
            logger.info(f"Model is already existed in {saved_model_path}")
            return

        url = f"{GIT_DOWNLOAD_URL}/{zipfile_name}.zip"
        saved_zipfile_path = os.path.join(saved_dir, f'{zipfile_name}.zip')
        cached_path(url, saved_zipfile_path, logger)
        
        with zipfile.ZipFile(saved_zipfile_path) as zip_file:
            zip_file.extractall(saved_dir)
        os.rename(
            Path(os.path.join(saved_dir, zipfile_name)),
            Path(saved_model_path),
        )
        os.remove(Path(saved_zipfile_path))

    @staticmethod
    def remove(args, logger):
        if args.student_name not in PRETRAINED_TOKENIZER_MAP.keys():
            logger.info(f"No matching distribution found for '{args.student_name}'")
            return
        cache_dir = os.path.join(
            CKPT_DIR, args.student_name, f"{args.training_mode}_{args.rm_accent_ratio}"
        )
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        logger.info("Model is removed.")