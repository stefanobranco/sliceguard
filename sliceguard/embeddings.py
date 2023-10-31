# Embedding support for text, images, audio
from multiprocess import set_start_method
import pandas as pd
import datasets
import numpy as np


def get_embedding_imports():
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoFeatureExtractor, AutoModel
        import torch
    except ImportError:
        raise Warning(
            'Optional dependency required! (pip install "sliceguard[embedding]")'
        )

    return SentenceTransformer, AutoFeatureExtractor, AutoModel, torch


def generate_text_embeddings(
    texts,
    model_name="all-MiniLM-L6-v2",
    hf_auth_token=None,
    hf_num_proc=None,
    hf_batch_size=1,
):
    SentenceTransformer, _, _, torch = get_embedding_imports()

    if hf_num_proc:
        print(
            "Warning: Multiprocessing cannot be used in generating text embeddings yet."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Embedding computation on {device} with batch size {hf_batch_size} and multiprocessing {hf_num_proc}."
    )

    model = SentenceTransformer(model_name, device=device, use_auth_token=hf_auth_token)
    embeddings = model.encode(texts, batch_size=hf_batch_size)
    return embeddings
