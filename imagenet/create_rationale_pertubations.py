import torch
import numpy as np
import PIL
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image
from .settings import Config


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def resize(img, scale, interp='nearest', diff=False):
    assert (len(img.shape) == 2)
    assert (interp == 'nearest')
    if diff:
        assert (int(img.shape[0] / scale) == img.shape[0] / scale)
        assert (int(img.shape[1] / scale) == img.shape[1] / scale)
        img_ = np.zeros((int(img.shape[0] / scale), int(img.shape[1] / scale)))
    else:
        assert (int(scale) == scale)
        img_ = np.zeros((scale * img.shape[0], scale * img.shape[1]))

    for i in range(img_.shape[0]):
        for j in range(img_.shape[1]):
            if diff:
                for r in range(scale):
                    for c in range(scale):
                        img_[i][j] += img[i * scale + r][j * scale + c]
            else:
                img_[i][j] = img[int(i / scale)][int(j / scale)]
    return img_


def create_blurred_circular_mask(mask_shape, radius, center=None, sigma=10):
    assert (len(mask_shape) == 2)
    if center is None:
        x_center = int(mask_shape[1] / float(2))
        y_center = int(mask_shape[0] / float(2))
        center = (x_center, y_center)
    y, x = np.ogrid[-y_center:mask_shape[0] - y_center, -x_center:mask_shape[1] - x_center]
    mask = x * x + y * y <= radius * radius
    grid = np.zeros(mask_shape)
    grid[mask] = 1
    if sigma is not None:
        grid = ndimage.filters.gaussian_filter(grid, sigma)
    return grid

def create_blurred_circular_mask_pyramid(mask_shape, radii, sigma=10):
    assert (len(mask_shape) == 2)
    num_masks = len(radii)
    masks = np.zeros((num_masks, 3, mask_shape[0], mask_shape[1]))
    for i in range(num_masks):
        masks[i, :, :, :] = create_blurred_circular_mask(mask_shape, radii[i], sigma=sigma)
    return masks

def find_initial_mask(model, image_tensor, blurred_tensor, class_index, original_prob, input_image_size):
    masks = create_blurred_circular_mask_pyramid((input_image_size, input_image_size), np.arange(0, int(np.round(input_image_size*0.78)), int(np.round(input_image_size/45))))
    masks = 1 - torch.tensor(masks, dtype=torch.float32, device="cuda")
    image_tensor = image_tensor.repeat(np.shape(masks)[0], 1, 1, 1)
    blurred_tensor = blurred_tensor.repeat(np.shape(masks)[0], 1, 1, 1)
    input_images = image_tensor * masks + blurred_tensor * (1 - masks)
    with torch.no_grad():
        probabilities = (torch.nn.Softmax(dim=1)(model(input_images))[:, class_index]).detach().cpu().numpy()

    percs = (probabilities - probabilities[-1]) / float(original_prob - probabilities[-1])
    try:
        first_i = np.where(percs < 0.01)[0][0]
    except:
        first_i = -1
    if input_image_size == 224:
        return np.array(Image.fromarray(masks[first_i][0].detach().cpu().numpy()).resize((28, 28), PIL.Image.Resampling.NEAREST))
    elif input_image_size == 384:
        return np.array(Image.fromarray(masks[first_i][0].detach().cpu().numpy()).resize((32, 32), PIL.Image.Resampling.NEAREST))
    else:
        raise Exception("Please implement mask size for this input image size.")

def tv(x, beta=3):
    d1 = np.zeros(x.shape)
    d2 = np.zeros(x.shape)
    d1[:-1, :] = np.diff(x, axis=0)
    d2[:, :-1] = np.diff(x, axis=1)
    v = np.sqrt(d1 * d1 + d2 * d2) ** beta
    e = v.sum()
    d1_ = (np.maximum(v, 1e-5) ** (2 * (beta / float(2) - 1) / float(beta))) * d1
    d2_ = (np.maximum(v, 1e-5) ** (2 * (beta / float(2) - 1) / float(beta))) * d2
    d11 = -d1_
    d22 = -d2_
    d11[1:, :] = -np.diff(d1_, axis=0)
    d22[:, 1:] = -np.diff(d2_, axis=1)
    dx = beta * (d11 + d22)
    return (e, dx)

# Implements mask creation as described in "Interpretable Explanations of Black Boxes by Meaningful Perturbation" by Fong et al.
# The code was mostly adapted from the original implementation to exactly replicate the original method, changes were made for the case of larger
# input image sizes (for the vision transformer), since the original hyperparameters were not suitable for this size.
def create_mask_pertubations(model, dataset, image_dict=None, plot=True, l1_coeff=1e-4, tv_coeff=1e-2, tv_beta=3, lr=1e-1, max_iter=300,
                             beta1=0.9, beta2=0.999):
    input_image_size = Config.input_image_size
    blurred_image_224, blurred_image_228 = dataset.get_blurred_image(image_dict["image_PIL_resized"], radius=10 if input_image_size == 224 else 20)

    with torch.no_grad():
        prob = torch.nn.Softmax(dim=1)(model(image_dict["image_tensor"]))
        pred_class = np.argmax(prob.cpu().data.numpy())
        label_prob = prob[0, image_dict['label']]
        print(f"Predicted class index: {pred_class} ({dataset.labels_to_class_names[pred_class]}). Correct class probability before perturbation: {label_prob}")

    mask = find_initial_mask(model, image_dict["image_tensor"], blurred_image_224, image_dict["label"], label_prob, input_image_size)

    m_t = np.zeros(mask.shape)
    v_t = np.zeros(mask.shape)

    if input_image_size == 224:
        resize_val = 8
    else:
        resize_val = 12

    last_weight_mean = 1
    weight_mean_diff_exp = None
    if input_image_size == 384:
        max_iter = 500
    for i in range(max_iter):
        upsampled_mask = resize(mask, resize_val)

        blurred_mask = torch.tensor(dataset.get_blurred_image(Image.fromarray(np.uint8(upsampled_mask*255)), radius=int(np.round(input_image_size/45)), return_array=True).reshape(1,1,input_image_size,input_image_size)/float(255), device="cuda", requires_grad=True, dtype=torch.float32)

        j_0 = np.random.randint(4)
        j_1 = np.random.randint(4)
        blurred_image_ = blurred_image_228[:, :, j_0:input_image_size + j_0, j_1:input_image_size + j_1]
        original_image_ = image_dict["image_tensor_plus_four"][:, :, j_0:input_image_size + j_0, j_1:input_image_size + j_1]

        input_image = blurred_mask * original_image_ + (1 - blurred_mask) * blurred_image_

        prediction = torch.nn.Softmax(dim=1)(model(input_image))[:, image_dict["label"]].mean()
        prediction.backward()

        l1_grad = l1_coeff * np.sign(mask - 1)
        tv_grad = tv_coeff * tv(mask, beta=tv_beta)[1]
        prediction_grad = blurred_mask.grad[0][0]
        prediction_grad = resize(prediction_grad, resize_val, diff=True)
        grad = prediction_grad + tv_grad + l1_grad

        m_t = beta1 * m_t + (1 - beta1) * grad
        v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)
        m_hat = m_t / float(1 - beta1 ** (i + 1))
        v_hat = v_t / float(1 - beta2 ** (i + 1))

        mask -= (float(lr) / (np.sqrt(v_hat) + 1e-8)) * m_hat

        mask[mask > 1] = 1
        mask[mask < 0] = 0
        blurred_mask.grad.zero_()

        m = np.mean(1 - mask)
        if i % 20 == 0:
            print(f"Probability for target class {prediction}. Mask avg: {m}")

        weight_mean_diff = last_weight_mean - m
        last_weight_mean = m
        if weight_mean_diff_exp is None:
            weight_mean_diff_exp = weight_mean_diff
        else:
            weight_mean_diff_exp = 0.95 * weight_mean_diff_exp + 0.05 * weight_mean_diff

        if weight_mean_diff_exp < 0.0002 and i >= 100 and m < 0.5:
            break

    upsampled_mask = resize(mask, resize_val)
    blurred_mask = 1-(np.expand_dims(dataset.get_blurred_image(Image.fromarray(np.uint8(upsampled_mask * 255)), radius=int(np.round(input_image_size/45)), return_array=True), axis=-1)/255)

    if plot:
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(image_dict["image_PIL_resized"])
        axarr[1].imshow((np.asarray(image_dict["image_PIL_resized"]).astype("float32") * blurred_mask).astype("uint8"))
        axarr[2].imshow(blurred_mask)

        plt.show()

    return blurred_mask