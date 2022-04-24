from inspect import istraceback
from cv2 import hconcat
import numpy as np
import tensorflow as tf
import os
import glob
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def get_images_paths(directory_name, image_type):
    """
    get the file name/path of all the files within a folder.
        e.g. glob.glob("/home/adam/*/*.txt").
    Use glob.escape to escape strings that are not meant to be patterns
        glob.glob(glob.escape(directory_name) + "/*.txt")

    :param directory_name: (str) the root directory name that contains all the images we want
    :param image: (str) either "jpg" or "png"
    :return: a list of queried files and directories
    """
    # concatnate strings
    end = "/*." + image_type

    return glob.glob(glob.escape(directory_name) + end)


def image_to_sketch(img, kernel_size=7):
    """
    Inputs:
    - img: RGB image, ndarray of shape []
    - kernel_size: 7 by default, used in DoG processing
    - greyscale: False by default, convert to greyscale image if True, RGB otherwise

    Returns:
    - RGB or greyscale sketch, ndarray of shape [] or []
    """
    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert
    inv = cv2.bitwise_not(grey)
    # blur
    blur = cv2.GaussianBlur(inv, (kernel_size, kernel_size), 0)
    # invert
    inv_blur = cv2.bitwise_not(blur)
    # convert to sketch
    sketch = cv2.divide(grey, inv_blur, scale=256.0)

    return sketch


# def visualize(img):
#     cv2.imwrite('sketch.png', img)
#     cv2.imshow('sketch image',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def extract_classwise_instances(samples, output_dir, label_field, size_lower_limit, ext=".png"):
    print("Extracting object instances...")
    for sample in samples.iter_samples(progress=True):
        img = cv2.imread(sample.filepath)
        img_h,img_w,c = img.shape
        if img_h >= size_lower_limit and img_w >= size_lower_limit:
            for det in sample[label_field].detections:
                mask = det.mask
                [x,y,w,h] = det.bounding_box
                x = int(x * img_w)
                y = int(y * img_h)
                h, w = mask.shape
                mask_img = img[y:y+h, x:x+w, :]
                alpha = mask.astype(np.uint8)*255
                alpha = np.expand_dims(alpha, 2)
                mask_img = np.concatenate((mask_img, alpha), axis=2)
                label = det.label
                label_dir = os.path.join(output_dir, label)

                if not os.path.exists(label_dir):
                    os.mkdir(label_dir)
                output_filepath = os.path.join(label_dir, det.id+ext)
                cv2.imwrite(output_filepath, mask_img)

def convert_dir_to_sketch(dir_path, save_dir, img_size, is_testing=False):
    # size: desired original img size, output will be twice as wide (concat)
    files = get_images_paths(dir_path, "png")
    i=0
    for f in files:
        ext = str(i)+".png"
        img = cv2.imread(f)
        img = pad_resize(img, img_size)
        if is_testing:
            out = cv2.hconcat([img, np.full((img_size, img_size, 3), (255,255,255), dtype=np.uint8)])
        else:
            sketch = image_to_sketch(img)
            out = cv2.hconcat([sketch, img])
        cv2.imwrite(os.path.join(save_dir, ext), out)
        i +=1


def pad_resize(img, img_size):
    # pad or resize img to square of side length (img_size)
    h, w, c = img.shape
    # white padding
    color = (255,255,255)

    out = 0

    if h > img_size or w > img_size:
        x_center = w // 2
        y_center = h // 2
        out = img[y_center - img_size // 2 : y_center + img_size // 2, x_center - img_size // 2 : x_center + img_size // 2]

    else:
        out = np.full((img_size, img_size, c), color, dtype=np.uint8)
        x_center = (img_size - w) // 2
        y_center = (img_size - h) // 2
        out[y_center: y_center + h, x_center: x_center + w] = img

    return out



def main():

    # get data
    dataset_name = "coco-image-example"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    label_field = "ground_truth"
    classes = ["dog","umbrella","truck"]
    
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["segmentations"],
        classes=classes,
        max_samples=100,
        label_field=label_field,
        dataset_name=dataset_name,
    )
    
    view = dataset.filter_labels(label_field, F("label").is_in(classes))

    output_dir = "/home/sli144/course/cs1470/final_project/dl_final_project/sample_data" 
    os.makedirs(output_dir,exist_ok=True)
    extract_classwise_instances(view, output_dir, label_field, 200)

    # generate sketches
    sketch_out_dir = "/home/sli144/course/cs1470/final_project/dl_final_project/sample_data/dog_sketch"
    os.makedirs(sketch_out_dir,exist_ok=True)
    convert_dir_to_sketch("/home/sli144/course/cs1470/final_project/dl_final_project/sample_data/dog",sketch_out_dir, 400)

if __name__ == '__main__':
    main()
