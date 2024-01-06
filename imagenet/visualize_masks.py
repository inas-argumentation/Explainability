import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from .models_and_data import ImageDataset
from .settings import Config, base_dir

# Plots all predicted masks for a dataset split for a specific method
def plot_predicted_masks(split, method):
    dataset = ImageDataset(split=split)

    for i in range(len(dataset)):
        sample = dataset[i]
        mask = np.load(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/{split}/{method}/{sample['name']}.npy"))
        plot_mask_and_bbox(sample["image_PIL_resized"], mask, sample["bbox_annotations"])

# Plots all predicted masks for a dataset split for all methods
def plot_predicted_mask_comparison_for_all_methods(split, methods):
    dataset = ImageDataset(split=split)

    images_per_row = int(np.ceil(len(methods) + 1) / 2)
    for i in range(len(dataset)):
        sample = dataset[i]

        f, axarr = plt.subplots(2, max(images_per_row, 2))
        axarr[0, 0].imshow(create_bbox_image_array(sample["image_PIL_resized"], sample["bbox_annotations"]))
        axarr[0, 0].set_title("Image")

        for pos, method in list(zip([(0, x) for x in range(1, images_per_row)], methods[:images_per_row-1])) + list(zip([(1, x) for x in range(images_per_row)], methods[images_per_row-1:])):
            mask = np.load(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/{split}/{method}/{sample['name']}.npy"))
            axarr[pos[0], pos[1]].imshow(np.asarray(np.reshape(mask, (Config.input_image_size, Config.input_image_size)) * 255, dtype="uint8"))
            axarr[pos[0], pos[1]].set_title(method)

        for ax_row in axarr:
            for ax in ax_row:
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        plt.show()

# Create an image with drawn bounding box
def create_bbox_image_array(image, bboxes):
    image_array = np.array(image)
    if bboxes is not None:
        for bbox in bboxes[2]:
            cv2.rectangle(image_array, bbox[:2], bbox[2:], (0, 0, 0), 2)
    return image_array

# Plots an image with corresponding bounding boxes and a given mask
def plot_mask_and_bbox(image, mask, bboxes=None):
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(np.asarray(image))
    axarr[1].imshow(np.asarray(np.reshape(mask, (Config.input_image_size, Config.input_image_size)) * 255, dtype="uint8"))
    axarr[2].imshow(create_bbox_image_array(image, bboxes))
    plt.show()

# Plots an image with corresponding bounding boxes
def plot_bbox(sample_dict):
    image_array = np.array(sample_dict["image_PIL_resized"])
    for bbox in sample_dict["bbox_annotations"][2]:
        cv2.rectangle(image_array, bbox[:2], bbox[2:], (200, 50, 50), 2)
    plt.axis('off')
    padding = sample_dict["padding_tensor"]
    plt.imshow(image_array[padding[1][0]:Config.input_image_size - padding[1][1], padding[0][0]:Config.input_image_size - padding[0][1]])
    #plt.savefig(f"figures/{name}.png", transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
