
import os, sys
import tarfile
import zipfile

from six.moves import urllib
import scipy.io

# -----------------------------Functions below are unused !!!-------------------------------
# Unused because the 2 functions below is to download a 500MB model used for normalization,
# which is time costing.
# Besides, the mean value of the model is calculated over 220 * 220 * 3,  different from the sizes 
# my network processes, so I think it's not useful. 
MODEL_URL = "http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat"

def get_model_data(dir_path="Model_zoo", model_url=MODEL_URL):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    print(filepath)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

def process_image(image, mean_pixel):
    return image - mean_pixel
