import numpy as np
import tensorflow as tf
import os
import glob
import cv2
# import fiftyone as fo
# import fiftyone.zoo as foz
# from fiftyone import ViewField as F


def get_images_paths(directory_name, image_type='png'):
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

def store_source_img(store_dir, size_lower_limit):
    dataset_name = "coco-image-example"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    label_field = "ground_truth"
    classes = ["dog"]

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["segmentations"],
        # classes=classes,
        # max_samples=10,
        label_field=label_field,
        dataset_name=dataset_name,
        shuffle=True,
    )

    # view = dataset.filter_labels(label_field, F("label").is_in(classes))
    os.makedirs(store_dir, exist_ok=True)
    extract_classwise_instances(dataset, store_dir, label_field, size_lower_limit)

def image_to_sketch(img, kernel_size=21):
    """
    Inputs:
    - img: RGB image, ndarray of shape []
    - kernel_size: 7 by default, used in DoG processing
    - greyscale: False by default, convert to greyscale image if True, RGB otherwise

    Returns:
    - RGB or greyscale sketch, ndarray of shape [] or []
    """

    # img = adjust_contrast(img)

    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert
    inv = cv2.bitwise_not(grey)
    # blur
    blur = cv2.GaussianBlur(inv, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
    # invert
    inv_blur = cv2.bitwise_not(blur)
    # convert to sketch
    sketch = cv2.divide(grey, inv_blur, scale=256.0)

    # sketch = adjust_contrast(sketch)

    out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    return out

# def dodgeV2(x,y):
#     return cv2.divide(x, 255-y, scale=256)

def adjust_contrast(img):

    # weight = np.ones(img.shape) * 1.2
    # out = np.uint8(cv2.multiply(np.float32(img), weight))

    out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

    return out

def pad_resize(img, img_size):
    # pad or resize img to square of side length (img_size)
    h, w, c = img.shape
    # white padding
    color = (255,255,255)

    out = None

    if h > img_size or w > img_size:
        down_points = (img_size, img_size)
        out = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)
    else:
        out = np.full((img_size, img_size, c), color, dtype=np.uint8)
        x_center = (img_size - w) // 2
        y_center = (img_size - h) // 2
        out[y_center: y_center + h, x_center: x_center + w] = img

    return out

def store_inputs(from_dir, to_dir, img_size, i):
    # store processed images (after concat)
    # size: desired original img size, output will be twice as wide (concat)
    files = get_images_paths(from_dir)
    for f in files:
        ext = str(i)+".png"
        img = cv2.imread(f)
        img = pad_resize(img, img_size)
        sketch = image_to_sketch(img)

        # img = pad_resize(img, img_size)
        # sketch = pad_resize(sketch, img_size)

        out = cv2.hconcat([sketch, img])
        cv2.imwrite(os.path.join(to_dir, ext), out)
        i+=1
    return i

def generate_data(from_dir, to_dir, img_size):
    # generate sketches
    os.makedirs(to_dir, exist_ok=True)

    folders = glob.glob(f'{from_dir}/*/')
    i=0
    for f in folders:
        print(f)
        i = store_inputs(f, to_dir, img_size,i)

def get_data(input_dir):
    input_paths = get_images_paths(input_dir)
    inputs = []

    for f in input_paths:
        inputs.append(cv2.imread(f))

    inputs = np.array(inputs, dtype='float32')

    return inputs

def main():
    # store_dir = "/home/sli144/course/cs1470/final_project/dl_final_project/sample_data"
    # store_dir = "sample_data"
    # size_lower_limit = 40
    # store_source_img(store_dir, size_lower_limit)

    from_dir = "/home/sli144/course/cs1470/final_project/Sketch-to-Image-Generation/sample_data/images"
    to_dir = "/home/sli144/course/cs1470/final_project/Sketch-to-Image-Generation/sample_data/sketches_grayscale"
    img_size = 64
    generate_data(from_dir, to_dir, img_size)


if __name__ == '__main__':
    main()
