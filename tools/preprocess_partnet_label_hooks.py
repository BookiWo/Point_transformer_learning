"""Optional label hooks for PartNet preprocessing.

Keep task-specific label parsing here so the main preprocessing script
focuses on point cloud IO, sampling, normalization, and output.
"""


def build_label_payload(file_path, config):
    """Return extra fields to be saved per sample.

    Current default is a no-op so preprocessing can run even when label
    conversion rules are not finalized.

    Args:
        file_path: Absolute path to source point cloud file.
        config: Parsed preprocess config dictionary.

    Returns:
        dict: Extra key-value pairs to merge into output payload.
    """
    _ = file_path
    _ = config
    return {}
