"""Evaluation datasets of the paper.

ID is held-out mathematics: the frozen pool test split (never trained on), GSM8K, Minerva, AIME 2024 and AMC 2024.
OOD is ARC-Challenge (lenient letter judge), SimpleQA (LLM judge) and MBPP+ (fenced-code test execution).
"""

ID = ("pool_heldout", "gsm8k", "minerva_math", "aime2024", "amc2024")
OOD = ("arc_challenge", "simpleqa", "mbpp_plus_fenced")
SUITES = {"id": ID, "ood": OOD}
