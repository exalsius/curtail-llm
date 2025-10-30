"""Medical Meadow dataset registry for medAlpaca integration.

This module defines all available Medical Meadow datasets with their sizes
and metadata for federated learning.
"""

# Medical Meadow dataset sizes (samples)
# Based on: https://github.com/kbressem/medAlpaca
MEDICAL_DATASETS = {
    # Curated combined dataset (medAlpaca-inspired, excluding USMLE for testing)
    "medalpaca/medical_meadow_curated": 226535,  # ~227K samples (excludes USMLE test set)

    # Core Medical Meadow datasets
    "medalpaca/medical_meadow_wikidoc": 67704,
    "medalpaca/medical_meadow_medical_flashcards": 33955,
    "medalpaca/medical_meadow_medqa": 10178,
    "medalpaca/medical_meadow_cord19": 13778,
    "medalpaca/medical_meadow_mmmlu": 3787,
    "medalpaca/medical_meadow_pubmed": 200000,  # Large corpus

    # Stack Exchange medical topics
    "medalpaca/medical_meadow_health_care_magic": 112165,
    "medalpaca/medical_meadow_stack_exchange_biology": 27326,
    "medalpaca/medical_meadow_stack_exchange_fitness": 9326,

    # Specialized medical datasets
    "medalpaca/medical_meadow_wikidoc_patient_information": 5942,
    "medalpaca/medical_meadow_mediqa": 2208,

    # USMLE (Medical licensing exams)
    "medalpaca/medical_meadow_usmle_self_assessment": 2903,
}


def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a medical dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "medalpaca/medical_meadow_wikidoc")

    Returns:
        Dictionary with dataset metadata
    """
    if dataset_name not in MEDICAL_DATASETS:
        raise ValueError(
            f"Unknown medical dataset: {dataset_name}. "
            f"Available datasets: {list(MEDICAL_DATASETS.keys())}"
        )

    return {
        "name": dataset_name,
        "size": MEDICAL_DATASETS[dataset_name],
        "split": get_default_splits(dataset_name),
    }


def get_default_splits(dataset_name: str) -> dict:
    """Get default train/validation splits for medical datasets.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with split names (e.g., {"train": "train", "validation": "validation"})
    """
    # Most Medical Meadow datasets use "train" split only
    # We'll need to split manually for validation
    return {
        "train": "train",
        "validation": None,  # Will split from train dynamically
    }


def list_available_datasets() -> list:
    """List all available medical datasets.

    Returns:
        List of dataset names
    """
    return sorted(MEDICAL_DATASETS.keys())


def get_total_samples(dataset_name: str) -> int:
    """Get total number of samples in a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Total number of samples
    """
    if dataset_name not in MEDICAL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return MEDICAL_DATASETS[dataset_name]
