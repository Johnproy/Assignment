import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms




# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "API_KEY_HERE"
TASK_ID = "01-mia"  #DONOT CHANGE



# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])




pub_ds.transform = transform
priv_ds.transform = transform


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


#membership submission

from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Creating final MIA submission...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()



def collect_logits(dataset, target_model, batch_size=128):
    def custom_collate(batch):
        ids = [x[0] for x in batch]
        imgs = torch.stack([x[1] for x in batch])
        labels = torch.tensor([x[2] for x in batch])
        if len(batch[0]) == 4:
            membership = [x[3] for x in batch]
            if membership[0] is None:
                return ids, imgs, labels
            membership = torch.tensor(membership)
            return ids, imgs, labels, membership
        return ids, imgs, labels

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )

    ids_all, labels_all, membership_all = [], [] ,[]
    logits_all = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                ids, imgs, labels, membership = batch
                membership_all.extend(membership.cpu().numpy())
            else:
                ids, imgs, labels = batch

            imgs = imgs.to(device)
            logits = target_model(imgs)

            logits_all.append(logits.cpu())
            labels_all.extend(labels.cpu().numpy())
            ids_all.extend([str(x) for x in ids])

    return (
        ids_all,
        torch.tensor(labels_all),
        np.array(membership_all),
        torch.cat(logits_all, dim=0)
    )

pub_ids, pub_labels, pub_mem, pub_logits = collect_logits(pub_ds, model)
priv_ids, priv_labels, _, priv_logits = collect_logits(priv_ds, model)

print("Public:", len(pub_ids), pub_logits.shape, pub_labels.shape, pub_mem.shape)
print("Private:", len(priv_ids), priv_logits.shape, priv_labels.shape)
print("Public membership values:", np.unique(pub_mem, return_counts=True))
print("Public label values:", np.unique(pub_labels.numpy(), return_counts=True))
print("Private label values:", np.unique(priv_labels.numpy(), return_counts=True))

assert len(pub_ids) == len(pub_labels) == len(pub_mem) == pub_logits.shape[0]
assert len(priv_ids) == len(priv_labels) == priv_logits.shape[0]
assert pub_logits.shape[1] == 9
assert priv_logits.shape[1] == 9
assert set(np.unique(pub_mem)).issubset({0, 1})
assert pub_labels.min().item() >= 0 and pub_labels.max().item() < 9
assert priv_labels.min().item() >= 0 and priv_labels.max().item() < 9

#LiRA




def tpr_at_5fpr(scores, membership):
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    membership = np.asarray(membership).astype(int)

    members = scores[membership == 1]
    nonmembers = scores[membership == 0]

    if len(members) == 0 or len(nonmembers) == 0:
        return 0.0, 0.0

    threshold = np.quantile(nonmembers, 0.95)
    tpr = np.mean(members >= threshold)
    fpr = np.mean(nonmembers >= threshold)
    return tpr, fpr

def make_resnet():
    m = resnet18(weights=None)
    m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc = torch.nn.Linear(512, 9)
    return m.to(device)

class MixedIndexDataset(torch.utils.data.Dataset):
    def __init__(self, pub_ds, priv_ds, pub_indices, priv_indices):
        self.pub_ds = pub_ds
        self.priv_ds = priv_ds
        self.items = [("pub", int(i)) for i in pub_indices] + [("priv", int(i)) for i in priv_indices]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, idx = self.items[i]
        if src == "pub":
            sample = self.pub_ds[idx]
        else:
            sample = self.priv_ds[idx]
        return sample[1], sample[2]


def train_shadow_model(pub_train_indices, priv_train_indices, epochs=15, batch_size=128):
    shadow = make_resnet()
    shadow.train()

    ds = MixedIndexDataset(pub_ds, priv_ds, pub_train_indices, priv_train_indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(shadow.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()
            loss = F.cross_entropy(shadow(imgs), labels)
            loss.backward()
            opt.step()

        scheduler.step()

    return shadow

print("\n--- Starting Likelihood Ratio Attack (LiRA) ---")
target_pub_loss = F.cross_entropy(pub_logits.to(device), pub_labels.to(device), reduction="none").cpu()
target_priv_loss = F.cross_entropy(priv_logits.to(device), priv_labels.to(device), reduction="none").cpu()

N_SHADOW = 128
SHADOW_EPOCHS = 15

shadow_pub_losses = []
shadow_priv_losses = []
shadow_pub_in_masks = []
shadow_priv_in_masks = []

rng_shadow = np.random.default_rng(12345)

pub_labels_for_shadow = pub_labels.numpy().astype(int)
priv_labels_for_shadow = priv_labels.numpy().astype(int)
unique_classes = np.unique(pub_labels_for_shadow)

for s in range(N_SHADOW):
    print(f"Training Mixed Shadow Model {s+1}/{N_SHADOW}...")

    pub_in_parts = []
    priv_in_parts = []

    for c in unique_classes:
        pub_idx_c = np.where(pub_labels_for_shadow == c)[0]
        priv_idx_c = np.where(priv_labels_for_shadow == c)[0]

        mem_c = pub_mem[pub_idx_c]
        p_c = float(np.mean(mem_c)) if len(mem_c) > 0 else float(np.mean(pub_mem))

        def sample_class_indices(idx_c, p_c, rng):
            idx_c = np.asarray(idx_c)

            if len(idx_c) == 0:
                return np.array([], dtype=int)
            if len(idx_c) == 1:
                return idx_c.copy()
            
            n_c = int(round(p_c * len(idx_c)))
            n_c = max(1, min(len(idx_c) - 1, n_c))
            return rng.choice(idx_c, size=n_c, replace=False)
        pub_chosen_c = sample_class_indices(pub_idx_c, p_c, rng_shadow)
        priv_chosen_c = sample_class_indices(priv_idx_c, p_c, rng_shadow)



        pub_in_parts.append(pub_chosen_c)
        priv_in_parts.append(priv_chosen_c)

    pub_in_idx = np.concatenate(pub_in_parts)
    priv_in_idx = np.concatenate(priv_in_parts)

    rng_shadow.shuffle(pub_in_idx)
    rng_shadow.shuffle(priv_in_idx)

    pub_in_mask = np.zeros(len(pub_ds), dtype=bool)
    priv_in_mask = np.zeros(len(priv_ds), dtype=bool)

    pub_in_mask[pub_in_idx] = True
    priv_in_mask[priv_in_idx] = True

    shadow_pub_in_masks.append(torch.tensor(pub_in_mask))
    shadow_priv_in_masks.append(torch.tensor(priv_in_mask))

    shadow_model = train_shadow_model(pub_in_idx, priv_in_idx, epochs=SHADOW_EPOCHS)
    shadow_model.eval()

    _, _, _, s_pub_logits = collect_logits(pub_ds, shadow_model)
    _, _, _, s_priv_logits = collect_logits(priv_ds, shadow_model)

    s_pub_loss = F.cross_entropy(
        s_pub_logits.to(device),
        pub_labels.to(device),
        reduction="none",
    ).cpu()

    s_priv_loss = F.cross_entropy(
        s_priv_logits.to(device),
        priv_labels.to(device),
        reduction="none",
    ).cpu()

    shadow_pub_losses.append(s_pub_loss)
    shadow_priv_losses.append(s_priv_loss)

    del shadow_model, s_pub_logits, s_priv_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\nComputing Sample-wise Gaussian LiRA Scores...")

shadow_pub_losses = torch.stack(shadow_pub_losses, dim=0)
shadow_priv_losses = torch.stack(shadow_priv_losses, dim=0)

shadow_pub_in_masks = torch.stack(shadow_pub_in_masks, dim=0).bool()
shadow_priv_in_masks = torch.stack(shadow_priv_in_masks, dim=0).bool()

def masked_mean_std(losses, masks):
    valid = masks.float()
    n = valid.sum(dim=0).clamp_min(1.0)

    mean = (losses * valid).sum(dim=0) / n

    var = (((losses - mean.unsqueeze(0)) ** 2) * valid).sum(dim=0)
    var = var / (n - 1).clamp_min(1.0)

    std = torch.sqrt(var + 1e-12)
    return mean, std

def torch_logpdf(x, mu, std):
    std = std + 1e-3
    return -0.5 * ((x - mu) / std) ** 2 - torch.log(std)


pub_in_mu, pub_in_std = masked_mean_std(shadow_pub_losses, shadow_pub_in_masks)
pub_out_mu, pub_out_std = masked_mean_std(shadow_pub_losses, ~shadow_pub_in_masks)

priv_in_mu, priv_in_std = masked_mean_std(shadow_priv_losses, shadow_priv_in_masks)
priv_out_mu, priv_out_std = masked_mean_std(shadow_priv_losses, ~shadow_priv_in_masks)

pub_shadow_raw = torch_logpdf(target_pub_loss, pub_in_mu, pub_in_std) - torch_logpdf(target_pub_loss, pub_out_mu, pub_out_std)
priv_shadow_raw = torch_logpdf(target_priv_loss, priv_in_mu, priv_in_std) - torch_logpdf(target_priv_loss, priv_out_mu, priv_out_std)


target_pub_log = torch.log(target_pub_loss + 1e-10)
target_priv_log = torch.log(target_priv_loss + 1e-10)

shadow_pub_log = torch.log(shadow_pub_losses + 1e-10)
shadow_priv_log = torch.log(shadow_priv_losses + 1e-10)

pub_in_mu, pub_in_std = masked_mean_std(shadow_pub_log, shadow_pub_in_masks)
pub_out_mu, pub_out_std = masked_mean_std(shadow_pub_log, ~shadow_pub_in_masks)

priv_in_mu, priv_in_std = masked_mean_std(shadow_priv_log, shadow_priv_in_masks)
priv_out_mu, priv_out_std = masked_mean_std(shadow_priv_log, ~shadow_priv_in_masks)

pub_shadow_log = torch_logpdf(target_pub_log, pub_in_mu, pub_in_std) - torch_logpdf(target_pub_log, pub_out_mu, pub_out_std)
priv_shadow_log = torch_logpdf(target_priv_log, priv_in_mu, priv_in_std) - torch_logpdf(target_priv_log, priv_out_mu, priv_out_std)

print("Mixed LiRA raw public TPR:", tpr_at_5fpr(pub_shadow_raw.numpy(), pub_mem))
print("Mixed LiRA log public TPR:", tpr_at_5fpr(pub_shadow_log.numpy(), pub_mem))
print("Mixed LiRA raw min/max:", float(pub_shadow_raw.min()), float(pub_shadow_raw.max()))
print("Mixed LiRA log min/max:", float(pub_shadow_log.min()), float(pub_shadow_log.max()))



def features_from_logits(logits, labels):
    labels = labels.long()
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)

    loss = F.cross_entropy(logits, labels, reduction="none")

    eps = 1e-7
    true_conf = probs[torch.arange(len(labels)), labels]
    true_conf = torch.clamp(true_conf, eps, 1 - eps)

    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1]

    entropy = -(probs * log_probs).sum(dim=1)

    one_hot = F.one_hot(labels, num_classes=probs.shape[1]).bool()
    log_one_minus = torch.log(torch.clamp(1.0 - probs, eps, 1.0))

    term_true = -(1.0 - true_conf) * torch.log(true_conf)
    term_other = -(probs * log_one_minus).masked_fill(one_hot, 0.0).sum(dim=1)

    modified_entropy = term_true + term_other

    return {
        "neg_loss": (-loss).numpy(),
        "true_conf": true_conf.numpy(),
        "logit_true": torch.log(true_conf / (1 - true_conf)).numpy(),
        "margin": (top1 - top2).numpy(),
        "neg_entropy": (-entropy).numpy(),
        "neg_modified_entropy": (-modified_entropy).numpy(),
    }


def log_gaussian_pdf(x, mu, sigma):
    sigma = sigma + 1e-8
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)


def gaussian_lira_score(train_vals, train_labels, train_mem, test_vals, test_labels):
    out = np.zeros_like(test_vals, dtype=float)

    for c in range(9):
        mem_vals = train_vals[(train_labels == c) & (train_mem == 1)]
        non_vals = train_vals[(train_labels == c) & (train_mem == 0)]

        if len(mem_vals) < 5 or len(non_vals) < 5:
            mem_vals = train_vals[train_mem == 1]
            non_vals = train_vals[train_mem == 0]

        mu_in, std_in = mem_vals.mean(), mem_vals.std() + 1e-8
        mu_out, std_out = non_vals.mean(), non_vals.std() + 1e-8

        idx = test_labels == c
        x = test_vals[idx]

        out[idx] = (
            log_gaussian_pdf(x, mu_in, std_in)
            - log_gaussian_pdf(x, mu_out, std_out)
        )

    return out


def rank01(x):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)

    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)

    return ranks / (len(x) - 1 + 1e-12)


def class_rank_score(values, labels):
    values = np.asarray(values)
    labels = np.asarray(labels)
    out = np.zeros_like(values, dtype=float)

    for c in np.unique(labels):
        idx = labels == c
        out[idx] = rank01(values[idx])

    return out


pub_labels_np = pub_labels.numpy()
priv_labels_np = priv_labels.numpy()

pub_feats = features_from_logits(pub_logits, pub_labels)
priv_feats = features_from_logits(priv_logits, priv_labels)

pub_target_logit = gaussian_lira_score(
    pub_feats["logit_true"],
    pub_labels_np,
    pub_mem,
    pub_feats["logit_true"],
    pub_labels_np,
)

priv_target_logit = gaussian_lira_score(
    pub_feats["logit_true"],
    pub_labels_np,
    pub_mem,
    priv_feats["logit_true"],
    priv_labels_np,
)

feature_names = [
    "neg_loss",
    "true_conf",
    "logit_true",
    "margin",
    "neg_entropy",
    "neg_modified_entropy",
]

extra_pub_gauss = {}
extra_priv_gauss = {}

for fname in feature_names:
    extra_pub_gauss[f"gauss_{fname}"] = gaussian_lira_score(
        pub_feats[fname],
        pub_labels_np,
        pub_mem,
        pub_feats[fname],
        pub_labels_np,
    )

    extra_priv_gauss[f"gauss_{fname}"] = gaussian_lira_score(
        pub_feats[fname],
        pub_labels_np,
        pub_mem,
        priv_feats[fname],
        priv_labels_np,
    )











def make_attack_matrix(feats, labels):
    labels = np.asarray(labels).astype(int)

    base = np.stack([feats[k] for k in feature_names], axis=1)
    base = np.nan_to_num(base, nan=0.0, posinf=50.0, neginf=-50.0)

    class_ranks = np.stack(
        [class_rank_score(feats[k], labels) for k in feature_names],
        axis=1,
    )

    onehot = np.eye(9)[labels]

    interactions = []
    for j in range(base.shape[1]):
        interactions.append(base[:, [j]] * onehot)
    interactions = np.concatenate(interactions, axis=1)

    X = np.concatenate(
        [
            base,
            base ** 2,
            class_ranks,
            onehot,
            interactions,
        ],
        axis=1,
    )

    return np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)


X_pub = make_attack_matrix(pub_feats, pub_labels_np)
X_priv = make_attack_matrix(priv_feats, priv_labels_np)
y_pub = pub_mem.astype(int)


def sigmoid_np(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def make_stratified_folds(y, n_splits=5, seed=2026):
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)

    folds = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)

        parts = np.array_split(idx, n_splits)
        for i in range(n_splits):
            folds[i].extend(parts[i].tolist())

    return [np.asarray(f, dtype=int) for f in folds]


def fit_lr_numpy(X_train, y_train, X_val, X_test, lr=0.05, n_iter=2500, l2=1e-4):
    y_train = y_train.astype(float)

    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    Xtr = (X_train - mu) / std
    Xva = (X_val - mu) / std
    Xte = (X_test - mu) / std

    n, d = Xtr.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    n_pos = max(np.sum(y_train == 1), 1)
    n_neg = max(np.sum(y_train == 0), 1)

    weight_pos = n / (2.0 * n_pos)
    weight_neg = n / (2.0 * n_neg)
    sample_w = np.where(y_train == 1, weight_pos, weight_neg)

    for _ in range(n_iter):
        p = sigmoid_np(Xtr @ w + b)
        err = (p - y_train) * sample_w

        grad_w = (Xtr.T @ err) / n + l2 * w
        grad_b = err.mean()

        w -= lr * grad_w
        b -= lr * grad_b

    val_score = sigmoid_np(Xva @ w + b)
    test_score = sigmoid_np(Xte @ w + b)

    return val_score, test_score


fold_groups = pub_labels_np.astype(int) * 2 + y_pub.astype(int)
folds = make_stratified_folds(fold_groups, n_splits=5, seed=2026)

pub_lr_score = np.zeros(len(y_pub), dtype=float)
priv_lr_folds = []

all_pub_idx = np.arange(len(y_pub))

for fold_id, va_idx in enumerate(folds):
    tr_mask = np.ones(len(y_pub), dtype=bool)
    tr_mask[va_idx] = False
    tr_idx = all_pub_idx[tr_mask]

    val_score, test_score = fit_lr_numpy(
        X_pub[tr_idx],
        y_pub[tr_idx],
        X_pub[va_idx],
        X_priv,
        lr=0.05,
        n_iter=2500,
        l2=1e-4,
    )

    pub_lr_score[va_idx] = val_score
    priv_lr_folds.append(test_score)

priv_lr_oof_ensemble = np.mean(priv_lr_folds, axis=0)

_, priv_lr_full = fit_lr_numpy(
    X_pub,
    y_pub,
    X_pub,
    X_priv,
    lr=0.05,
    n_iter=3500,
    l2=1e-4,
)

priv_lr_score = 0.5 * priv_lr_oof_ensemble + 0.5 * priv_lr_full

print("OOF NumPy LR public TPR:", tpr_at_5fpr(pub_lr_score, pub_mem))


pub_components = {
    "shadow_raw": rank01(pub_shadow_raw.numpy()),
    "shadow_log": rank01(pub_shadow_log.numpy()),
    "target_logit": rank01(pub_target_logit),
    "lr_target": rank01(pub_lr_score),
    "class_neg_loss": class_rank_score(pub_feats["neg_loss"], pub_labels_np),
    "class_logit_true": class_rank_score(pub_feats["logit_true"], pub_labels_np),
    "neg_loss": rank01(pub_feats["neg_loss"]),
    "neg_entropy": rank01(pub_feats["neg_entropy"]),
    "neg_modified_entropy": rank01(pub_feats["neg_modified_entropy"]),
}

priv_components = {
    "shadow_raw": rank01(priv_shadow_raw.numpy()),
    "shadow_log": rank01(priv_shadow_log.numpy()),
    "target_logit": rank01(priv_target_logit),
    "lr_target": rank01(priv_lr_score),
    "class_neg_loss": class_rank_score(priv_feats["neg_loss"], priv_labels_np),
    "class_logit_true": class_rank_score(priv_feats["logit_true"], priv_labels_np),
    "neg_loss": rank01(priv_feats["neg_loss"]),
    "neg_entropy": rank01(priv_feats["neg_entropy"]),
    "neg_modified_entropy": rank01(priv_feats["neg_modified_entropy"]),
}



for k in extra_pub_gauss:
    pub_components[k] = rank01(extra_pub_gauss[k])
    priv_components[k] = rank01(extra_priv_gauss[k])

shadow_meta_sources = {
    "shadow_raw_meta": (pub_shadow_raw.numpy(), priv_shadow_raw.numpy()),
    "shadow_log_meta": (pub_shadow_log.numpy(), priv_shadow_log.numpy()),
    "lr_meta": (pub_lr_score, priv_lr_score),
}

for name, (pub_v, priv_v) in shadow_meta_sources.items():
    pub_meta = gaussian_lira_score(
        pub_v,
        pub_labels_np,
        pub_mem,
        pub_v,
        pub_labels_np,
    )

    priv_meta = gaussian_lira_score(
        pub_v,
        pub_labels_np,
        pub_mem,
        priv_v,
        priv_labels_np,
    )

    pub_components[name] = rank01(pub_meta)
    priv_components[name] = rank01(priv_meta)    

for k in list(pub_components.keys()):
    tpr_normal, _ = tpr_at_5fpr(pub_components[k], pub_mem)
    tpr_flip, _ = tpr_at_5fpr(1.0 - pub_components[k], pub_mem)

    if tpr_flip > tpr_normal:
        print("FLIPPING", k, "normal:", tpr_normal, "flipped:", tpr_flip)
        pub_components[k] = 1.0 - pub_components[k]
        priv_components[k] = 1.0 - priv_components[k]



candidate_scores = {}


for k in pub_components:
    candidate_scores[f"single_{k}"] = (pub_components[k], priv_components[k])

#3way blend
for a in np.linspace(0, 1, 11):
    for b in np.linspace(0, 1 - a, 11):
        c = 1.0 - a - b

        name = f"blend_raw{a:.2f}_log{b:.2f}_target{c:.2f}"

        pub_mix = (
            a * pub_components["shadow_raw"]
            + b * pub_components["shadow_log"]
            + c * pub_components["target_logit"]
        )

        priv_mix = (
            a * priv_components["shadow_raw"]
            + b * priv_components["shadow_log"]
            + c * priv_components["target_logit"]
        )

        candidate_scores[name] = (pub_mix, priv_mix)


blend_keys = [
    "shadow_log",
    "shadow_raw",
    "target_logit",
    "lr_target",
    "class_logit_true",
    "class_neg_loss",
    "gauss_neg_loss",
    "gauss_logit_true",
    "gauss_true_conf",
    "gauss_margin",
    "gauss_neg_entropy",
    "gauss_neg_modified_entropy",
    "shadow_raw_meta",
    "shadow_log_meta",
    "lr_meta",
]
for k1 in blend_keys:
    for k2 in blend_keys:
        if k1 >= k2:
            continue

        for w in np.linspace(0, 1, 11):
            name = f"blend_{k1}_{w:.2f}_{k2}_{1-w:.2f}"

            pub_mix = w * pub_components[k1] + (1 - w) * pub_components[k2]
            priv_mix = w * priv_components[k1] + (1 - w) * priv_components[k2]

            candidate_scores[name] = (pub_mix, priv_mix)



single_tprs = {}
for k, v in pub_components.items():
    tpr, fpr = tpr_at_5fpr(v, pub_mem)
    single_tprs[k] = tpr

top_keys = sorted(single_tprs, key=single_tprs.get, reverse=True)[:8]
print("TOP COMPONENTS:", [(k, single_tprs[k]) for k in top_keys])


for m in range(2, min(8, len(top_keys)) + 1):
    keys = top_keys[:m]

    pub_mix = np.mean([pub_components[k] for k in keys], axis=0)
    priv_mix = np.mean([priv_components[k] for k in keys], axis=0)

    candidate_scores[f"top{m}_mean"] = (pub_mix, priv_mix)


rng = np.random.default_rng(2026)

for r in range(2000):
    keys = top_keys
    weights = rng.dirichlet(np.ones(len(keys)) * 0.5)

    pub_mix = np.zeros_like(pub_components[keys[0]], dtype=float)
    priv_mix = np.zeros_like(priv_components[keys[0]], dtype=float)

    for w, k in zip(weights, keys):
        pub_mix += w * pub_components[k]
        priv_mix += w * priv_components[k]

    candidate_scores[f"random_top_blend_{r}"] = (pub_mix, priv_mix)            


for k, v in pub_components.items():
    tpr, fpr = tpr_at_5fpr(v, pub_mem)
    print("SINGLE", k, "TPR:", tpr, "FPR:", fpr)

def robust_public_score(scores, membership, n_boot=80, seed=777):
    scores = np.asarray(scores)
    membership = np.asarray(membership).astype(int)

    full_tpr, full_fpr = tpr_at_5fpr(scores, membership)

    rng = np.random.default_rng(seed)
    boot_tprs = []

    n = len(scores)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        mem_b = membership[idx]

        if np.sum(mem_b == 1) == 0 or np.sum(mem_b == 0) == 0:
            continue

        tpr_b, _ = tpr_at_5fpr(scores[idx], mem_b)
        boot_tprs.append(tpr_b)

    boot_tprs = np.asarray(boot_tprs)

    if len(boot_tprs) == 0:
        return full_tpr, full_tpr, 0.0

    boot_mean = boot_tprs.mean()
    boot_std = boot_tprs.std()

    
    objective = 0.70 * full_tpr + 0.30 * boot_mean - 0.15 * boot_std

    return objective, full_tpr, boot_std


print("\nBlend validation:")

raw_results = []

for name, (pub_s, priv_s) in candidate_scores.items():
    tpr, fpr = tpr_at_5fpr(pub_s, pub_mem)
    raw_results.append((tpr, name, pub_s, priv_s))
    print(name, "TPR@5FPR:", tpr, "FPR:", fpr)

raw_results.sort(reverse=True, key=lambda x: x[0])


top_raw = raw_results[:100]

best_name = None
best_obj = -1
best_tpr = -1

print("\nRobust validation on top candidates:")

for raw_tpr, name, pub_s, priv_s in top_raw:
    obj, tpr, std = robust_public_score(pub_s, pub_mem, n_boot=40)

    print(name, "raw_tpr:", raw_tpr, "robust_obj:", obj, "boot_std:", std)

    if obj > best_obj:
        best_obj = obj
        best_tpr = tpr
        best_name = name

print("BEST BLEND:", best_name, "public TPR:", best_tpr, "robust obj:", best_obj)
print("Using final blend:", best_name)

raw_best_tpr, raw_best_name, _, _ = raw_results[0]

print("RAW BEST:", raw_best_name, raw_best_tpr)
print("ROBUST BEST:", best_name, best_tpr, best_obj)



if best_tpr >= raw_best_tpr - 0.0015:
    final_name = best_name
else:
    final_name = raw_best_name

print("FINAL SELECTED:", final_name)

scores = rank01(candidate_scores[final_name][1])

df = pd.DataFrame({
    "id": priv_ids,
    "score": scores,
})

df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:", OUTPUT_CSV)
print("Rows:", len(df))
print("Score min/max:", df["score"].min(), df["score"].max())
print("Duplicate IDs:", df["id"].duplicated().sum())











# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args, _ = parser.parse_known_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)

