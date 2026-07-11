"""Model-level utilities.

Provides helper functions for freezing PLM layers during fine-tuning.
"""
import torch

def freeze_bert_layers(plm: torch.nn.Module, freeze_layers: int) -> None:
    """
    Freezes the specified number of BERT layers.
    Args:
        plm: The pre-trained language model.
        freeze_layers (int): Number of layers to freeze.
    """
    if freeze_layers <= 0:
        return

    print(f"Freezing BERT embeddings and first {freeze_layers} encoder layers.")

    # Determine base model attribute (e.g., .bert, .distilbert, .roberta)
    base_model = None
    for attr in ["bert", "distilbert", "roberta", "electra", "albert"]:
        if hasattr(plm, attr):
            base_model = getattr(plm, attr)
            break

    if base_model is None:
        # Fallback: some models might use 'base_model' or just be the model themselves
        base_model = getattr(plm, "base_model", plm)

    # 1. Freeze Embeddings
    if hasattr(base_model, "embeddings"):
        for param in base_model.embeddings.parameters():
            param.requires_grad = False

    # 2. Freeze Encoder Layers
    layers = None
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
        layers = base_model.encoder.layer
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "layer"):
        layers = base_model.transformer.layer
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "blocks"):
        layers = base_model.transformer.blocks

    if layers is not None:
        for i in range(freeze_layers):
            if i < len(layers):
                for param in layers[i].parameters():
                    param.requires_grad = False
            else:
                print(f"Warning: requested freezing layer {i} but model only has {len(layers)} layers.")
    else:
        print(f"Warning: could not find layers to freeze for model architecture {type(base_model)}")
