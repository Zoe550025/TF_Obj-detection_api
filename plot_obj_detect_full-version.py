import os
import matplotlib

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here

# specify which device you want to work on.
# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"]="0" # TODO: specify your computational device

import tensorflow as tf # import tensorflow

# checking that GPU is found
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# other import
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm



import sys # importyng sys in order to access scripts located in a different folder

path2scripts = 'E:/tzuwen/tree_segmentation2/TF_Obj-dection_api/models/research/' # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import

# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
matplotlib.use('TkAgg')

# NOTE: your current working directory should be Tensorflow.

# TODO: specify two pathes: to the pipeline.config file and to the folder with trained model.
path2config ='E:/tzuwen/tree_segmentation2/TF_Obj-dection_api/workspace/training_demo/exported-models/my_model_step400000/pipeline.config'
path2model = 'E:/tzuwen/tree_segmentation2/TF_Obj-dection_api/workspace/training_demo/exported-models/my_model_step400000/checkpoint/'

# do not change anything in this cell
configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()

path2label_map = 'E:/tzuwen/tree_segmentation2/TF_Obj-dection_api/workspace/training_demo/annotations/labelmap.pbtxt' # TODO: provide a path to the label map file
category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)

def test_folder(folderpath):
    try:
        os.makedirs(folderpath)
    # 檔案已存在的例外處理
    except FileExistsError:
        print("檔案已存在。")

def detect_fn(image):
    """
    Detect objects in image.

    Args:
      image: (tf.tensor): 4D input image

    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """

    return np.array(Image.open(path))

import json
def inference_with_plot(path2images,fname,root_path,box_th=0.25):
    """
    Function that performs inference and plots resulting b-boxes

    Args:
      path2images: an array with pathes to images
      fname： an array with images' name
      box_th: (float) value that defines threshold for model prediction.
      root_path： an array with pathes which images are stored.

    Returns:
      None
    """
    for image_path,image_name,stored_path in zip(path2images,fname,root_path):
        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,               # TODO:oringnal = 200
            min_score_thresh=box_th,
            agnostic_mode=False,
            line_thickness=5)

        coordinates = viz_utils.return_coordinates(
            image_np_with_detections,
            np.squeeze(detections['detection_boxes']),
            np.squeeze(detections['detection_classes']).astype(np.int32),
            np.squeeze(detections['detection_scores']),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=5,
            min_score_thresh=0.0)                  # TODO:oringnal = 0.20

        plt.figure(figsize=(8.32,8.32),dpi=100)   # TODO:調整尺寸，這裡是英吋
        plt.imshow(image_np_with_detections)
        plt.axis('off')
        print('Done')
        plt.savefig(stored_path, bbox_inches='tight',pad_inches = 0)
        plt.close()

        filename_string = stored_path.replace(image_name,"")+"coordinates"
        textfile = open(filename_string + ".json", "a")
        textfile.write(json.dumps(coordinates))
        textfile.write("\n")

from pathlib import Path
rootdir = "E:\\tzuwen\\tree_segmentation2\\resize_test\\640_20220624_2-19\\2022\\6"#'E:\\tzuwen\\tree_segmentation2\\resize_test\\640_0-56'       # TODO:要框出來的目錄
pic_list = []
pic_name = []
stored_path_list = []
for root, subFolders, files in os.walk(rootdir):
    for pic in files:
        pic_name.append(pic)
        f = os.path.join(root,pic)
        pic_list.append(f)
        test_folder(f.replace(pic,"")+"results")
        tmp = f.replace(pic,"")+"results\\"+str(pic)
        stored_path_list.append(tmp)
#print(stored_path_list)

#print(pic_name)
inference_with_plot(pic_list,pic_name,stored_path_list)


