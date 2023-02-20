import sys
import ast
from typing import List
import torch
import numpy as np
from pyemd import emd
from transformers import pipeline


class StyleIntensityClassifier:
    """
    Utility for calculating Style Transfer Intensity between
    two pieces of text (i.e. input and output of TST model)
    This custom evaluation metric aims to quantify the magnitude of transfered
    style between two texts. To accomplish this, we pass input and output texts
    through a trained style classifier to produce two distributions. We then
    utilize Earth Movers Distance (EMD) to calculate the minimum "cost"/"work"
    required to turn the input distribution into the output distribution. This
    metric allows us to capture a more nuanced, per-example measure of style
    transfer when compared to simply applying a classifier to determine the
    binary decision of achieved target style? (yes/no).
    Attributes:
        model_identifier (str)
    """

    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        self._build_pipeline()

    def _build_pipeline(self):

        self.pipeline = pipeline(
            task="text-classification",
            model=self.model_identifier,
            device=self.device,
            return_all_scores=True,
        )

    def score(self, input_text: List[str]):
        """
        Classify a given input text as subjective or neutral using
        model initialized by the class.
        Args:
            input_text (`str` or `List[str]`) - Input text for classification
        Returns:
            classification (dict) - a dictionary containing the label, score, and
                distribution between classes
        """
        result = self.pipeline(input_text)
        distributions = np.array(
            [[label["score"] for label in item] for item in result]
        )
        return [
            {
                "label": self.pipeline.model.config.id2label[scores.argmax()],
                "score": scores.max(),
                "distribution": scores.tolist(),
            }
            for scores in distributions
        ]

    def calculate_transfer_intensity(
        self, input_text: List[str], output_text: List[str], target_class_idx: int = 1
    ) -> List[float]:
        """
        Calcualates the style transfer intensity (STI) between two pieces of text.
        Args:
            input_text (list) - list of input texts with indicies corresponding
                to counterpart in output_text
            ouptput_text (list) - list of output texts with indicies corresponding
                to counterpart in input_text
            target_class_idx (int) - index of the target style class used for directional
                score correction
        Returns:
            A list of floats with corresponding style transfer intensity scores.
        """

        if len(input_text) != len(output_text):
            raise ValueError(
                "input_text and output_text must be of same length with corresponding items"
            )

        input_dist = [item["distribution"] for item in self.score(input_text)]
        output_dist = [item["distribution"] for item in self.score(output_text)]

        return [
            self.calculate_emd(input_dist[i], output_dist[i], target_class_idx)
            for i in range(len(input_dist))
        ]

    @staticmethod
    def calculate_emd(input_dist, output_dist, target_class_idx):
        """
        Calculate the direction-corrected Earth Mover's Distance (aka Wasserstein distance)
        between two distributions of equal length. Here we penalize the EMD score if
        the output text style moved further away from the target style.
        Reference: https://github.com/passeul/style-transfer-model-evaluation/blob/master/code/style_transfer_intensity.py
        Args:
            input_dist (list) - probabilities assigned to the style classes
                from the input text to style transfer model
            output_dist (list) - probabilities assigned to the style classes
                from the outut text of the style transfer model
        Returns:
            emd (float) - Earth Movers Distance between the two distributions
        """

        N = len(input_dist)
        distance_matrix = np.ones((N, N))
        dist = emd(np.array(input_dist), np.array(output_dist), distance_matrix)

        transfer_direction_correction = (
            1 if output_dist[target_class_idx] >= input_dist[target_class_idx] else -1
        )

        return round(dist * transfer_direction_correction, 4)


def StyleEvaluator(sc, corpus):
    return sc.score(corpus)


if __name__ == "__main__":
    corpus = sys.argv[1]
    split = sys.argv[2]
    label = sys.argv[3]
    gen_file = f"{corpus}/{split}.{label}"
    model_path = "checkpoint-100000"
    sc = StyleIntensityClassifier(model_identifier=model_path)

    with open(gen_file) as f:
        gen = [line.strip() for line in f]

    su = 0
    scores_src = []

    # For Tgt in Valid, Test
    if split in ["test", "valid"] and label == "tgt":
        for sentencelist in gen:
            sl = ast.literal_eval(sentencelist)
            for sentence in sl:
                score_src = StyleEvaluator(sc, [sentence])
                su += score_src[0]["score"]
                scores_src.append(score_src)
    else:
        for sentence in gen:
            score_src = StyleEvaluator(sc, [sentence])
            su += score_src[0]["score"]
            scores_src.append(score_src)

    print("Average Style Strength:", su / len(scores_src))
    output_file = f"scores/style_{corpus}_{split}_{label}.txt"
    output = scores_src
    with open(output_file, "w") as f:
        print(output, file=f)
        print("Average Style Strength:", su / len(scores_src), file=f)
