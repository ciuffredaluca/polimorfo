from PIL import Image
from functools import partial
import requests
from io import BytesIO
import multiprocessing
from multiprocessing.dummy import Pool
from tqdm import tqdm_notebook as tqdm
import logging

log = logging.getLogger(__name__)

def download_image(url_path):
    """download an image and save to the destination path

    Arguments:
        url {str} -- the url of the image
        path {Path} -- the base path where the images shold be saved

    Keyword Arguments:
        timeout {int} -- the timeout in seconds (default: {1})
    """
    url, path = url_path
    if path.exists():
        return
    try:
        response = requests.get(url, timeout=15, allow_redirects=False)
        if response.ok:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.save(path, "JPEG", optimize=True)
        response.close()
    except Exception as ex:
        log.debug('error processing the url %s' % (url), ex)


def process_images(urls_filepath, timeout):
    parallelism = multiprocessing.cpu_count() // 2

    with Pool(parallelism) as pool:
        with tqdm(total=len(urls_filepath), desc='download images') as pbar:
            for _ in pool.imap_unordered(download_image, urls_filepath):
                pbar.update()