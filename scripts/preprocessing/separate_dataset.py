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


test_list_name = "test_list.txt"
images = []
img_path = []
separate_images = []

def readFile(path):
    img_list = []
    with open(path) as f:
        for line in f.readlines():
            img_list.append(line)
    return img_list

def iterate_dir(source, dest, separate, copy_xml):
    # source = source.replace('\\', '/')
    # dest = dest.replace('\\', '/')
    # separate = separate.replace('\\', '/')
    count = 0
    test_dir = os.path.join(dest, 'test')
    f = open(os.path.join(dest,test_list_name), 'w')

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for root,subfolder, files in os.walk(source):
        for file in files:
            if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', file):
                file = file.replace(" ","").replace("\n","")
                images.append(file)
                img_path.append(root+"\\"+ file)
    #print(images)
    print(len(img_path))

    #read separate list
    separate_images = readFile(separate)
    num_separate_images = len(separate_images)
    #print(separate_images)

    for i in range(num_separate_images):
        fname = os.path.basename(separate_images[i].replace(" ","").replace("\n","").replace("\r",""))
        # copyfile(os.path.join(source, fname),
        #           os.path.join(test_dir,fname))
        # if copy_xml:
        #     xml_filename = os.path.splitext(fname)[0]+".xml"
        #     # copyfile(os.path.join(source,xml_filename),
        #     #          os.path.join(test_dir,xml_filename))
        try:
            images.remove(fname.replace(" ","").replace("\n",""))
            img_path.remove(separate_images[i].replace("\n","").replace("\r",""))
            print(separate_images[i].replace("\n",""), "removed.")
        except:
            print(separate_images[i], "not found")

    for filename in img_path:
        img_name = os.path.basename(filename)
        img_name_plain = img_name.replace(".png","")
        if os.path.isfile(os.path.join(test_dir,img_name)):
            count+=1
            img_name = img_name_plain + "_" + str(count) + ".png"
        try:
            copyfile(filename, os.path.join(test_dir, img_name))
        except:
            print(filename, "can't copy")
        #write test data path into txt
        f.write(filename+"\n")
        # if copy_xml:
        #     xml_filename = os.path.splitext(filename)[0]+'.xml'
        #     copyfile(os.path.join(source, xml_filename),
        #              os.path.join(test_dir, xml_filename))
    print(len(img_path))
    f.close()

def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
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
        '-s', '--separateList',
        help='Path to the folder where the image dataset need to separate. If not specified, the CWD will be used.',
        type=str,
        default = None
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
    iterate_dir(args.imageDir, args.outputDir, args.separateList, args.xml)
    #python separate_dataset.py -i "E:\tzuwen\tree_segmentation2\data" -o "E:\tzuwen\tree_segmentation2\data_separated_2" -s "E:\tzuwen\tree_segmentation2\data_separated_2\train_list.txt"

if __name__ == '__main__':
    main()