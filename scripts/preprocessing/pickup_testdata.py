""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]

separate dataset of images into training and testing sets, if you have one of dataset

optional arguments:
  -h, --help            show this help message and exit

  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -s SEPARATELIST, --separateList SEPARATElIST
                        Path to the list where the image dataset need to separate. If not specified, the CWD will be used.'
"""
import os
import re
from shutil import copyfile
import argparse


test_list_name = "test_list.txt"
images = []
img_path = []
separate_images = []

def readFile(path):
    img_list = []
    with open(path) as f:
        for line in f.readlines():
            img_list.append(line.replace("\n",""))
    return img_list

def copypng(list_path,test_dir):
    List = readFile(list_path)
    count = 0
    for l in List:
        img_name = os.path.basename(l)
        img_name_plain = img_name.replace(".png", "")
        if os.path.isfile(os.path.join(test_dir, img_name)):
            count += 1
            img_name = img_name_plain + "_" + str(count) + ".png"
        try:
            copyfile(l, os.path.join(test_dir, img_name))
        except:
            print(l, "can't copy")


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-s', '--separateList',
        help='Path to the folder where the image dataset need to separate. If not specified, the CWD will be used.',
        type=str,
        default = None
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )

    args = parser.parse_args()

    # if args.outputDir is None:
    #     args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    copypng(args.separateList, args.outputDir)
    #python pickup_testdata.py -s "E:\tzuwen\tree_segmentation2\data_separated_2\test_list.txt" -o "E:\tzuwen\tree_segmentation2\data_separated_2\test"

if __name__ == '__main__':
    main()