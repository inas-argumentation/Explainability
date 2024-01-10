import os.path

from movie_reviews import train_classifier, create_rationales, run_evaluations, settings

def run_experiment():

    # Masks from the original experiments are included. To use them, set "save_name" to "MaRC_paper" (uncomment to do so).
    settings.set_save_name("MaRC_paper")
    # Enable the legacy mode to get exact reproductions of the scores from the paper. See the comment at settings.set_legacy_mode() for details.
    settings.set_legacy_mode(True)

    # Train model if no checkpoint exists
    if not os.path.exists(os.path.join(settings.base_dir, f"saved_models/clf_{settings.Config.save_name}.pkl")):
        train_classifier.train()

    # Create rationales for all interpretability methods
    create_rationales.create_rationales_for_complete_data_set_and_all_interpretability_approaches(split="test")

    # Evaluate results
    for method in settings.interpretability_approaches:
        if method is None:
            continue
        print(f"\nMethod: {method}")

        print("\nFaithfulness evaluation:")
        run_evaluations.evaluate_faithfulness(method)

        # The agreement with human annotations is done with respect to a "target_score", which determines the score that
        # is used to train the kernel regression model for thresholding. For each target score, all three score results are
        # reported, but the score reported in the paper is always the score that is produced with the matching target score.
        print(f"\nAgreement evaluation (target score: F1):")
        run_evaluations.evaluate_predictions(method, target_score="F1")

        print(f"\nAgreement evaluation (target score: IoU):")
        run_evaluations.evaluate_predictions(method, target_score="IoU")

        print(f"\nAgreement evaluation (target score: mAP):")
        run_evaluations.evaluate_predictions(method, target_score="mAP")



if __name__ == '__main__':
    run_experiment()