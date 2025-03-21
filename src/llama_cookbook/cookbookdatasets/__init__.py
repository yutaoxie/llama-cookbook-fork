# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

def get_grammar_dataset(*args, **kwargs):
    from .grammar_dataset.grammar_dataset import get_dataset
    return get_dataset(*args, **kwargs)

def get_alpaca_dataset(*args, **kwargs):
    from .alpaca_dataset import InstructionDataset
    return InstructionDataset(*args, **kwargs)

def get_samsum_dataset(*args, **kwargs):
    from .samsum_dataset import get_preprocessed_samsum
    return get_preprocessed_samsum(*args, **kwargs)

def get_llamaguard_toxicchat_dataset(*args, **kwargs):
    from .toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_toxic
    return get_toxic(*args, **kwargs)

def get_custom_dataset(*args, **kwargs):
    from .custom_dataset import get_custom_dataset as get_custom
    return get_custom(*args, **kwargs)

def get_data_collator(*args, **kwargs):
    from .custom_dataset import get_data_collator as get_collator
    return get_collator(*args, **kwargs)

DATASET_PREPROC = {
    "alpaca_dataset": get_alpaca_dataset,
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
}

DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator
}

__all__ = [
    'DATASET_PREPROC',
    'DATALOADER_COLLATE_FUNC',
]
