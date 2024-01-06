import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from .models_and_data import load_model, ImageDataset
from .settings import Config, base_dir

def evaluate_faithfulness(method, split):
    model = load_model()

    # Add slight randomness to predicted masks to act as tiebreaker if many pixels have the same score
    np.random.seed(0)
    mask_randomness = np.random.randn(Config.input_image_size, Config.input_image_size) * 1e-5

    dataset = ImageDataset(split)

    sufficiency_values = []
    comprehensiveness_values = []

    # For the faithfulness evaluation, we evaluate with respect to all four logical choices of uninformative background.
    backgrounds = [dataset.get_white_image().detach().cpu().numpy(), dataset.get_black_image().detach().cpu().numpy()]
    backgrounds.append(np.zeros_like(backgrounds[0]))

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating", file=sys.stdout):
            sample_dict = dataset[i]
            p = sample_dict["padding_tensor"]

            try:
                mask = np.load(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/{split}/{method}/{sample_dict['name']}.npy")).reshape((Config.input_image_size, Config.input_image_size))
            except:
                print(f"{sample_dict['name']} is missing")
                continue

            # Average over multiple choices for uninformative background input
            backgrounds = [dataset.get_blurred_image(sample_dict["image_PIL_resized"])[0].detach().cpu().numpy(),
                           dataset.get_black_image().detach().cpu().numpy(),
                           dataset.get_mean_image().detach().cpu().numpy(),
                           dataset.get_white_image().detach().cpu().numpy()]

            mask = (mask + mask_randomness)[p[1][0]:Config.input_image_size - p[1][1], p[0][0]:Config.input_image_size - p[0][1]]

            # Sort mask scores to determine the threshold (e.g., 10% of highest scoring pixels).
            sorted_scores = np.sort(np.reshape(mask, (-1,)))[::-1]

            # Perform sufficiency and comprehensiveness test for current sample
            for test in ["sufficiency", "comprehensiveness"]:
                current_values = []

                # Average over different selection thresholds
                for percentage in range(0, 105, 5):
                    different_background_values = []

                    if percentage == 0:
                        score_threshold = sorted_scores[0] + 1e-4
                    elif percentage == 100:
                        score_threshold = sorted_scores[-1] - 1e-4
                    else:
                        score_threshold = sorted_scores[int(np.round(len(sorted_scores)*(percentage/100)))]

                    if test == "sufficiency":
                        selection = mask < score_threshold
                    else:
                        selection = mask >= score_threshold
                    # Create mask from the selected indices
                    indices = np.expand_dims(np.tile(np.expand_dims(np.pad(selection, [p[1], p[0]], constant_values=True), axis=0), (3, 1, 1)), axis=0)

                    # Uncomment to see selection
                    #plt.imshow(indices[0, 0])
                    #plt.show()

                    for background in backgrounds:
                        image = sample_dict["image_tensor"].detach().cpu().numpy()
                        # Replace masked pixels with current background.
                        image[indices] = background[indices]

                        image_tensor = torch.tensor(image).to("cuda")
                        prediction = torch.softmax(model(image_tensor), dim=-1).detach().cpu().numpy()[0, sample_dict["label"]]
                        different_background_values.append(prediction)
                    # Total value is the average over the different backgrounds.
                    current_values.append(np.mean(different_background_values))

                if test == "sufficiency":
                    sufficiency_values.append(np.mean([current_values[-1] - c for c in current_values[1:-1]]))
                elif test == "comprehensiveness":
                    comprehensiveness_values.append(np.mean([current_values[0] - c for c in current_values[1:-1]]))

    print(f"Sufficiency: {np.mean(sufficiency_values)}")
    print(f"Comprehensiveness: {np.mean(comprehensiveness_values)}")
