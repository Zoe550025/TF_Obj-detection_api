""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]

separate dataset of images into training and testing sets, if you have one of dataset

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -s SEPARATELIST, --separateList SEPARATElIST
                        Path to the list where the image dataset need to separate. If not specified, the CWD will be used.'
  -x, --xml             Set this flag if you want the xml annotation files to be processed and copied over.
"""


import os
import re
from shutil import copyfile
import argparse

def test_folder(folderpath):
    try:
        os.makedirs(folderpath)
    # 檔案已存在的例外處理
    except FileExistsError:
        print("檔案已存在。")

def iterate_dir(source, dest):
    img_path = []
    for root,subfolder, files in os.walk(source):
        for file in files:
            if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', file):
                file = file.replace(" ","").replace("\n","")
                img_path.append(root+"\\"+ file)
    #print(img_path)
    for p in img_path:
        # p = E:\\tzuwen\\tree_segmentation2\\data\\0\\0-0\\2021-01-27_1552.png
        p_list = p.split("\\")
        filename = p_list[6]
        year = filename[:4]
        month = int(filename[5:7])

        temp_addr = p_list[4]+"\\"+p_list[5]+"\\"+ year+ "\\" + str(month) + "\\" + filename
        final_addr = os.path.join(dest,temp_addr)
        test_folder(final_addr.replace(filename,""))
        copyfile(p,final_addr)
        print(temp_addr,"done")


    print("done")



def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images by month",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )

    parser.add_argument(
        '-x', '--xml',
        help='Set this flag if you want the xml annotation files to be processed and copied over.',
        action='store_true'
    )
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir)
    #python .\separate_data_by_month.py -i E:\tzuwen\tree_segmentation2\data -o E:\tzuwen\tree_segmentation2\data_by_month


if __name__ == '__main__':
    main()

