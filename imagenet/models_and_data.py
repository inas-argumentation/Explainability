from .settings import Config, model_types, base_dir
from torchvision import models, transforms
from PIL import ImageFilter, Image
from torchvision.transforms import functional as FT
from torch.utils.data import Dataset
import numpy as np
import json
import os
import torch
import untangle
import cv2

# Allows to convert cuda tensor to numpy by typing t.np()
def to_np(self):
    return self.detach().cpu().numpy()
setattr(torch.Tensor, "np", to_np)

def load_model(type=None):
    if type is None:
        type = Config.model_type
    if type == model_types[0]:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to("cuda")
    elif type == model_types[1]:
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1).to("cuda")
    elif type == model_types[2]:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1).to("cuda")
    else:
        raise Exception("Unknown model type!")
    model.eval()
    return model

# Add black borders to create square image
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = (max_wh - w) / 2
        vp = (max_wh - h) / 2
        padding = (int(np.floor(hp)), int(np.floor(vp)), int(np.ceil(hp)), int(np.ceil(vp)))
        return FT.pad(image, padding, 0, 'constant')

def load_class_names():
    with open(os.path.join(base_dir, "data/imagenet/imagenet_class_index.json"), "r") as f:
        class_names = json.load(f)
    return class_names

def load_blacklist():
    with open(os.path.join(base_dir, "data/imagenet/validation_blacklist.txt"), "r") as f:
        blacklist = [int(x) for x in f.read().split("\n") if len(x) > 0]
    return blacklist

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def listdir(folder):
    return [x for x in os.listdir(folder) if x != ".gitignore"]

# The dataset to load the images. Implements four different splits: "train" and "val" for hyperparameter tuning, "faith"
# for the faithfulness evaluation (random selection of 500 "val" images) and "figure" (the images used to generate the figure in our paper).
class ImageDataset(Dataset):
    def __init__(self, split="figure"):
        self.split = split

        self.labels_to_class_names = {int(x): y for x, y in load_class_names().items()}
        self.class_codes_to_labels = {**{y[0]: int(x) for x, y in self.labels_to_class_names.items()},
                                      **{y[1].lower(): int(x) for x, y in self.labels_to_class_names.items()}}
        self.set_split(split)

        # Transform image array into tensor as neural network input
        self.transform_list = transforms.Compose([
                               SquarePad(),
                               transforms.Resize(Config.input_image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        # Same transforms, but for "perturbations" approach by Fong et al., which requires slightly larger image (four pixels wider and higher)
        self.transform_list_plus_4 = transforms.Compose([
            SquarePad(),
            transforms.Resize(Config.input_image_size + 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    # Load filenames and labels for given dataset split
    def set_split(self, split):
        if split not in ["val", "train", "figure", "faith"]:
            raise Exception("Dataset split not available.")
        self.split = split
        if split == "train":
            self.filenames = sorted(listdir(os.path.join(base_dir, f"data/imagenet/images/{split}")))
            self.gt = {x: self.class_codes_to_labels[x[:x.index("_")]] for x in self.filenames}
        elif split == "val":
            self.filenames = sorted(listdir(os.path.join(base_dir, f"data/imagenet/images/{split}")))
            blacklist = load_blacklist()
            self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if i+1 not in blacklist]
            self.gt = {y[0]: int(y[1][:-1]) for y in [x.split(" ") for x in open(os.path.join(base_dir, "data/val_gt.txt"), "r")]}
        elif split == "faith":
            with open(os.path.join(base_dir, "data/imagenet/faithfulness_500_sample.json"), "r") as f:
                self.filenames = json.loads(f.read())
            self.gt = {y[0]: int(y[1][:-1]) for y in [x.split(" ") for x in open(os.path.join(base_dir, "data/imagenet/val_gt.txt"), "r")]}
        elif split == "figure":
            self.filenames = sorted(listdir(os.path.join(base_dir, f"data/imagenet/images/{split}")))
            self.gt = {x: self.class_codes_to_labels[x[:x.index("_")]] for x in self.filenames}


    def transform(self, image, plus_four=False):
        if plus_four:
            # Transform to image with correct input size plus four pixels
            return self.transform_list_plus_4(image)
        # Transform to image with correct input size
        return self.transform_list(image)

    def transform_unsqueeze(self, image, plus_four=False):
        return torch.unsqueeze(self.transform(image, plus_four=plus_four), dim=0).to("cuda")

    # Perform inverse transform to make tensor displayable
    def inverse_normalize(self, image):
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        inv_tensor = inv_normalize(image)
        return inv_tensor

    def load_annotation_data(self, name):
        xml = untangle.parse(os.path.join(base_dir, f"data/imagenet/annotations/{self.split if self.split != 'faith' else 'val'}/{name[:-5]}.xml"))
        if type(xml.annotation.object) == list:
            objects = xml.annotation.object
        else:
            objects = [xml.annotation.object]
        width = int(xml.annotation.size.width.cdata)
        height = int(xml.annotation.size.height.cdata)
        ratio = Config.input_image_size / (max_wh := max(width, height))
        hp = int(np.floor((max_wh - width) / 2 * ratio))
        vp = int(np.floor((max_wh - height) / 2 * ratio))
        labels = []
        class_names = []
        bboxes = []
        for object in objects:
            labels.append(self.class_codes_to_labels[object.name.cdata.lower()])
            class_names.append(self.labels_to_class_names[labels[-1]][1])
            coordinates = [object.bndbox.xmin.cdata, object.bndbox.ymin.cdata,
                           object.bndbox.xmax.cdata, object.bndbox.ymax.cdata]
            coordinates = [int(np.round((float(x) * ratio))) for x in coordinates]
            coordinates[0] += hp
            coordinates[2] += hp
            coordinates[1] += vp
            coordinates[3] += vp
            bboxes.append(coordinates)
        return labels, class_names, bboxes

    def __len__(self):
        return len(self.filenames)

    # Compiles different versions of the image used as input for different approaches, together with other information
    def __getitem__(self, idx):
        d = {}
        d["name"] = self.filenames[idx]
        d["label"] = self.gt[d["name"]]
        d["class_name"] = self.labels_to_class_names[d["label"]][1]
        d["image_PIL_original"] = pil_loader(os.path.join(base_dir, f"data/imagenet/images/{self.split if self.split != 'faith' else 'val'}/{d['name']}"))
        d["image_tensor"] = self.transform_unsqueeze(d["image_PIL_original"])
        d["image_tensor_plus_four"] = self.transform_unsqueeze(d["image_PIL_original"], True)
        width, height = d["image_PIL_original"].width, d["image_PIL_original"].height
        size = max(width, height)
        padded_image = Image.new('RGB', (size, size), (0, 0, 0))
        x = (size - width) / 2
        y = (size - height) / 2
        d["padding_original"] = [(int(np.ceil(x)), int(np.floor(x))), (int(np.ceil(y)), int(np.floor(y)))]
        scale = Config.input_image_size / size
        d["padding_tensor"] = [(int(np.ceil(x*scale)), int(np.floor(x*scale))), (int(np.ceil(y*scale)), int(np.floor(y*scale)))]
        padded_image.paste(d["image_PIL_original"], (d["padding_original"][0][0], d["padding_original"][1][0]))
        d["image_PIL_resized"] = padded_image.resize((Config.input_image_size, Config.input_image_size))
        try:
            d["bbox_annotations"] = self.load_annotation_data(d["name"])
        except:
            d["bbox_annotations"] = [[], [], []]
        return d

    # Allows displaying of image specified by idx
    def show(self, idx):
        name = self.filenames[idx]
        gt_label = self.gt[name]
        gt_class_name = self.labels_to_class_names[gt_label][1]
        image = pil_loader(os.path.join(base_dir, f"data/imagenet/images/{self.split if self.split != 'faith' else 'val'}/{name}"))
        image_array = np.array(image)
        if self.split != "figure":
            labels, class_names, bboxes = self.load_annotation_data(name)
            for box in bboxes:
                cv2.rectangle(image_array, box[:2], box[2:], (0, 0, 0), 3)
            print(name, gt_label, gt_class_name, labels, class_names)
        Image.fromarray(image_array).show()

    # Used by some approaches to get black image as background
    def get_black_image(self, noise=False, sigma=0.03):
        if noise:
            image = np.ones((Config.input_image_size, Config.input_image_size, 3), dtype="float32") * 0.1 + \
                    np.random.randn(Config.input_image_size * Config.input_image_size * 3).reshape((Config.input_image_size, Config.input_image_size, 3)) * sigma
            image[image < 0] = 0
            image = (image*255).astype("uint8")
        else:
            image = np.zeros((Config.input_image_size, Config.input_image_size, 3), dtype="uint8")
        image = Image.fromarray(image)
        image = self.transform_unsqueeze(image).to("cuda")
        return image

    # Used by some approaches to get white image as background
    def get_white_image(self, noise=False, sigma=0.03):
        if noise:
            image = np.ones((Config.input_image_size, Config.input_image_size, 3), dtype="float32") * 0.9 + \
                    np.random.randn(Config.input_image_size * Config.input_image_size * 3).reshape((Config.input_image_size, Config.input_image_size, 3)) * sigma
            image[image > 1] = 1
            image = (image*255).astype("uint8")
        else:
            image = np.ones((Config.input_image_size, Config.input_image_size, 3), dtype="uint8") * 255
        image = Image.fromarray(image)
        image = self.transform_unsqueeze(image).to("cuda")
        return image

    # Used by some approaches to get grey image as background
    def get_mean_image(self, noise=False, sigma=0.03):
        if noise:
            image = torch.tensor(np.random.randn(Config.input_image_size * Config.input_image_size * 3).reshape((1, 3, Config.input_image_size, Config.input_image_size)) * sigma, dtype=torch.float32)
            return image.to("cuda")
        else:
            return torch.zeros((1, 3, Config.input_image_size, Config.input_image_size), dtype=torch.float32).to("cuda")

    # Used by some approaches to get a blurred image as background
    def get_blurred_image(self, pil_image, radius=10, return_array=False):
        image = pil_image.filter(ImageFilter.GaussianBlur(radius))
        if return_array:
            return np.array(image)
        return self.transform_unsqueeze(image, plus_four=False), self.transform_unsqueeze(image, plus_four=True)

    # Used by some approaches to get random noise image as background
    def get_random_noise_image(self, sigma):
        image = (np.random.random((Config.input_image_size, Config.input_image_size, 3)) * sigma * 255 + 127.5).astype("uint8")
        image = Image.fromarray(image)
        image = self.transform_unsqueeze(image)
        return image