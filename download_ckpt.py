import gdown
import os
import zipfile

ZIP_PATH = "/tmp/east_lite_ckpt.zip"
CKPT_PATH = "/tmp/ckpt"


def load_ckpt():
    if not os.path.exists(ZIP_PATH):
        url = "https://drive.google.com/uc?id=1_SyIM-CNTBqdPsviw2aVa7qXjavjtRY0"
        gdown.download(url, ZIP_PATH, quiet=False)

    if not os.path.exists(CKPT_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(CKPT_PATH)
