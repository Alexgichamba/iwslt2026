"""Evaluation metrics for ASR and MT."""

from __future__ import annotations


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Word Error Rate using jiwer."""
    from jiwer import wer
    return wer(references, predictions)


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """BLEU score using sacrebleu."""
    import sacrebleu
    result = sacrebleu.corpus_bleu(predictions, [references])
    return {"bleu": result.score, "detail": str(result)}


def compute_chrf(predictions: list[str], references: list[str]) -> dict:
    """chrF++ score using sacrebleu."""
    import sacrebleu
    result = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    return {"chrf": result.score, "detail": str(result)}
