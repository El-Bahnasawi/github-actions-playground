from typing import Literal


Label = Literal["neutral", "toxic"]


def score_text(text: str) -> float:
    """
    Fake "ML" scoring function.
    The longer the text, the more "toxic" we pretend it is.
    Obviously nonsense, but the structure matches a real model.
    """
    length = len(text.strip())
    if length == 0:
        return 0.0

    # Normalize to [0, 1]
    score = min(length / 100, 1.0)
    return score


def predict_label(text: str) -> Label:
    score = score_text(text)
    return "toxic" if score >= 0.5 else "neutral"
