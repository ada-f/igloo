from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
SPECIAL_TOKENS = ["[CDRs]", "[CDRe]"]
SECTIONS = ['FW1', 'CDR1', 'FW2', 'CDR2', 'FW3', 'CDR4', 'FW4', 'CDR3', 'FW5']