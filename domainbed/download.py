from torchvision.datasets import MNIST
from zipfile import ZipFile
import argparse
import tarfile
import gdown
import os


# utils #######################################################################

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


# MNIST #######################################################################

def download_mnist(data_dir):
    # Original URL: http://yann.lecun.com/exdb/mnist/
    full_path = stage_path(data_dir, "MNIST")
    MNIST(data_dir, download=True)


# VLCS ########################################################################

def download_vlcs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "VLCS")

    download_and_extract("https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8",
                         os.path.join(data_dir, "VLCS.tar.gz"))


# PACS ########################################################################

def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)


# Office-Home #################################################################

def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "OfficeHome")

    download_and_extract("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
                         os.path.join(data_dir, "office_home.zip"))

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, required=True, choices=['RotatedMNIST', 'VLCS', 'PACS', 'OfficeHome'])
    args = parser.parse_args()

    if args.dataset == 'RotatedMNIST':
        download_mnist(args.data_dir)
    elif args.dataset == 'VLCS':
        download_vlcs(args.data_dir)
    elif args.dataset == 'PACS':
        download_pacs(args.data_dir)
    elif args.dataset == 'OfficeHome':
        download_office_home(args.data_dir)
    else:
        raise NotImplementedError("Dataset not found: {}".format(args.dataset))

    print("Done!")