import os.path
from imagenet.settings import base_dir, model_types, Config
from imagenet import create_rationales, visualize_masks, evaluation, settings

def perform_faithfulness_experiment():
    # Three model types were tested (and two included in the paper). See settings.py for details.
    for current_model_type in model_types[1:]:
        print(f"\nPerforming faithfulness experiment for {current_model_type} model\n")
        settings.set_model_type(current_model_type)

        # The original masks from the predictions are included. The save_name needs to be set to "run_0" in imagenet/settings.py to use the existing masks.
        settings.set_save_name("run_0")

        # Predict masks for all methods implemented in this repo
        create_rationales.generate_masks_for_whole_dataset_and_all_methods(split="faith")

        # Additional methods were tested using other existing repos, results are included in this repo.
        additional_methods = ["GradCam", "ExcitationBP"] if current_model_type == model_types[1] else ["Attribution", "GradCam", "RawAttention", "Rollout", "TrAttMaps"]

        # Run evaluation
        for method in Config.explainability_methods + additional_methods:
            if os.path.exists(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/faith/{method}")): # Skip methods with no data available
                print(f"\nMethod: {method}")
                evaluation.evaluate_faithfulness(method, "faith")

def generate_figure_images():
    # Three model types were tested (and two included in the paper). See settings.py for details.
    for current_model_type in model_types[1:]:
        print(f"\nCreating figures for {current_model_type} model\n")
        settings.set_model_type(current_model_type)

        # The original masks from the predictions are included. The save_name needs to be set to "run_0" in imagenet/settings.py to use the existing masks.
        settings.set_save_name("run_0")

        # Predict masks for all methods implemented in this repo
        create_rationales.generate_masks_for_whole_dataset_and_all_methods(split="figure")

        # Additional methods were tested using other existing repos, results are included in this repo.
        additional_methods = ["GradCam", "ExcitationBP"] if current_model_type == model_types[1] else ["Attribution", "GradCam", "RawAttention", "Rollout", "TrAttMaps"]

        available_methods = [x for x in (Config.explainability_methods + additional_methods) if os.path.exists(os.path.join(settings.base_dir, f"predictions/{Config.save_name}/{Config.model_type}/figure/{x}"))]
        visualize_masks.plot_predicted_mask_comparison_for_all_methods("figure", available_methods)

if __name__ == '__main__':
    generate_figure_images()
    perform_faithfulness_experiment()