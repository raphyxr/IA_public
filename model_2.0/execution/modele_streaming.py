import torch
import torch.nn as nn

# Minimal subset of functions required by mod.py (modele)
VOCAB_LIMIT = 100000
EMB_DIM = 64
HIDDEN_DIM = 128


def embedding(ids, emb_layer):
    """Transforme des IDs (torch.long) en vecteurs d'embedding."""
    return emb_layer(ids)


def init_training_objects(vocab_size=VOCAB_LIMIT, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, lr=0.05, optimizer_name="sgd"):
    """Cree le modele + loss + optimizer."""
    emb_layer = nn.Embedding(vocab_size, emb_dim)
    lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
    fc = nn.Linear(hidden_dim, vocab_size)
    criterion = nn.CrossEntropyLoss()

    params = list(emb_layer.parameters()) + list(lstm.parameters()) + list(fc.parameters())
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        optimizer = torch.optim.SGD(params, lr=lr)

    return emb_layer, lstm, fc, optimizer, criterion


def load_checkpoint_if_compatible(path, emb_layer, lstm, fc, optimizer=None, vocab_size=None, emb_dim=None, hidden_dim=None):
    """Charge un checkpoint seulement si les tailles correspondent, sinon leve une erreur claire."""
    ckpt = torch.load(path, map_location="cpu")

    meta = ckpt.get("meta")
    if isinstance(meta, dict):
        expected = {}
        if vocab_size is not None:
            expected["vocab_size"] = vocab_size
        if emb_dim is not None:
            expected["emb_dim"] = emb_dim
        if hidden_dim is not None:
            expected["hidden_dim"] = hidden_dim

        mismatches = []
        for k, v in expected.items():
            if meta.get(k) != v:
                mismatches.append(f"{k}: ckpt={meta.get(k)} vs actuel={v}")
        if mismatches:
            raise ValueError("checkpoint incompatible (tailles differentes). " + " | ".join(mismatches))
    else:
        emb_w = ckpt.get("emb", {}).get("weight")
        fc_w = ckpt.get("fc", {}).get("weight")
        if emb_w is not None and tuple(emb_w.shape) != tuple(emb_layer.weight.shape):
            raise ValueError(f"checkpoint incompatible (embedding): ckpt={tuple(emb_w.shape)} vs actuel={tuple(emb_layer.weight.shape)}")
        if fc_w is not None and tuple(fc_w.shape) != tuple(fc.weight.shape):
            raise ValueError(f"checkpoint incompatible (fc): ckpt={tuple(fc_w.shape)} vs actuel={tuple(fc.weight.shape)}")

    emb_layer.load_state_dict(ckpt["emb"])
    lstm.load_state_dict(ckpt["lstm"])
    fc.load_state_dict(ckpt["fc"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
