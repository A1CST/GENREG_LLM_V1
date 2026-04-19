#!/usr/bin/env python3
"""Generate expressive embedding visualizations for the evolved 768-dim
embedding space. Produces:

  assets/embedding_tsne.png                2D t-SNE with 11 semantic clusters
  assets/embedding_3d.png                  3D PCA (for comparison, updated)
  assets/embedding_3d_front.png            3D PCA from a different angle
  assets/embedding_3d_side.png             3D PCA from a third angle
  assets/embedding_centroid_heatmap.png    pairwise cosine between
                                            category mean embeddings
"""
import os, sys, pickle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model import GenregLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Semantic categories — hand-picked words we know should cluster
CATEGORIES = {
    "countries": ["france", "germany", "italy", "spain", "japan", "china",
                   "russia", "brazil", "canada", "mexico", "india",
                   "australia", "poland", "greece", "egypt", "turkey",
                   "vietnam", "nigeria", "argentina", "ireland"],
    "verbs": ["running", "walking", "talking", "playing", "thinking",
              "eating", "drinking", "writing", "reading", "singing",
              "dancing", "fighting", "waiting", "looking", "working",
              "helping", "buying", "selling", "driving", "flying"],
    "numbers": ["one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten", "eleven", "twelve", "twenty",
                 "thirty", "forty", "fifty", "hundred", "thousand"],
    "years": ["1990", "1995", "2000", "2005", "2008", "2009", "2010",
               "2011", "2012", "2013", "2014", "2015", "2016", "2017",
               "2018", "2019", "2020"],
    "colors": ["red", "blue", "green", "yellow", "black", "white",
                "purple", "orange", "pink", "brown", "gray", "silver",
                "gold", "violet"],
    "royalty": ["king", "queen", "prince", "princess", "duke", "duchess",
                 "emperor", "empress", "lord", "lady", "knight",
                 "baron", "crown", "throne", "royal", "noble"],
    "science": ["physics", "chemistry", "biology", "mathematics",
                 "astronomy", "geology", "psychology", "sociology",
                 "genetics", "ecology", "neuroscience", "electron",
                 "proton", "molecule", "atom", "theorem"],
    "sports": ["football", "basketball", "baseball", "soccer", "tennis",
                "hockey", "cricket", "rugby", "golf", "boxing",
                "swimming", "cycling", "racing", "skating", "surfing"],
    "time": ["monday", "tuesday", "wednesday", "thursday", "friday",
              "saturday", "sunday", "january", "february", "march",
              "april", "june", "july", "august", "september",
              "october", "november", "december", "morning",
              "evening", "night"],
    "body": ["head", "eye", "ear", "nose", "mouth", "arm", "leg",
              "hand", "foot", "finger", "heart", "brain", "bone",
              "blood", "skin", "muscle", "stomach", "liver"],
    "emotions": ["happy", "sad", "angry", "fear", "love", "hate",
                  "joy", "sorrow", "pride", "shame", "hope", "despair",
                  "surprise", "disgust", "anger", "grief",
                  "jealousy", "wonder"],
}

CATEGORY_COLORS = {
    "countries": "#e41a1c",   # red
    "verbs":     "#377eb8",   # blue
    "numbers":   "#4daf4a",   # green
    "years":     "#984ea3",   # purple
    "colors":    "#ff7f00",   # orange
    "royalty":   "#ffff33",   # yellow
    "science":   "#a65628",   # brown
    "sports":    "#f781bf",   # pink
    "time":      "#999999",   # gray
    "body":      "#00ced1",   # teal
    "emotions":  "#ff1493",   # magenta
}


def main():
    print("loading model + embedding...", flush=True)
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")
    m = GenregLM(ckpt_dir, device="cuda")
    V = m.V
    print(f"  V = {V:,}", flush=True)

    # Compute embedding for all tokens
    print("embedding all tokens...", flush=True)
    emb_table = m.embed.embed(torch.arange(V, device="cuda"))
    emb_np = emb_table.cpu().numpy()
    print(f"  shape: {emb_np.shape}", flush=True)

    # Pick the background cloud: top-3000 word tokens by ID frequency
    # (ids 96..95+3000 represent the most common words post-char region)
    # But we also want the category words present.
    bg_ids = list(range(96, min(96 + 3000, V)))

    # Map category words → ids
    cat_ids = {}
    cat_labels = {}
    for cat, words in CATEGORIES.items():
        cat_ids[cat] = []
        for w in words:
            tid = m.token_to_id.get(w)
            if tid is not None and tid != m.token_to_id.get("<unk>", 1):
                cat_ids[cat].append(tid)
                cat_labels[tid] = cat
        print(f"  {cat}: {len(cat_ids[cat])} tokens in vocab", flush=True)

    # Combine all ids to visualize
    all_cat_ids = []
    for cat, ids in cat_ids.items():
        all_cat_ids.extend(ids)
    all_cat_ids = list(set(all_cat_ids))
    viz_ids = sorted(set(bg_ids) | set(all_cat_ids))
    viz_emb = emb_np[viz_ids]
    print(f"  viz matrix: {viz_emb.shape}", flush=True)

    # -------- t-SNE (2D, local-structure preserving) --------
    print("running t-SNE (this takes ~30s)...", flush=True)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                random_state=42, init="pca", max_iter=1000)
    emb_2d = tsne.fit_transform(viz_emb)
    print(f"  t-SNE done: {emb_2d.shape}", flush=True)
    id_to_row = {tid: r for r, tid in enumerate(viz_ids)}

    # -------- PCA 3D (global-variance preserving) --------
    print("running PCA...", flush=True)
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(viz_emb)
    var_ratio = pca.explained_variance_ratio_
    print(f"  PCA variance ratios: {var_ratio}", flush=True)

    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # ----- Chart 1: t-SNE with semantic clusters -----
    print("plotting t-SNE...", flush=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    # Background (grey, small, semi-transparent)
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                c="#d0d0d0", s=6, alpha=0.35, linewidths=0,
                label=f"{len(viz_ids):,} frequent tokens")
    # Each category on top with its own color
    for cat, ids in cat_ids.items():
        if not ids:
            continue
        rows = [id_to_row[i] for i in ids if i in id_to_row]
        if not rows:
            continue
        rows = np.array(rows)
        ax.scatter(emb_2d[rows, 0], emb_2d[rows, 1],
                    c=CATEGORY_COLORS[cat], s=70,
                    edgecolors="black", linewidths=0.5,
                    alpha=0.9, label=f"{cat} ({len(ids)})")
        # Annotate a few representative tokens per category
        for i, r in enumerate(rows[:4]):
            tok_str = m.id_to_token.get(ids[i], "?")
            ax.annotate(tok_str, (emb_2d[r, 0], emb_2d[r, 1]),
                         fontsize=7, alpha=0.75,
                         xytext=(4, 2), textcoords="offset points")
    ax.set_title("Evolved embedding — t-SNE 2D\n"
                  "(PPMI-SVD init + evolved skip+encoder, gradient-free)",
                  fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
               ncol=1, bbox_to_anchor=(1.23, 1))
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = os.path.join(assets_dir, "embedding_tsne.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}", flush=True)

    # ----- Chart 2, 3, 4: 3D PCA from three angles -----
    angles = [(25, -60, "embedding_3d.png"),
              (15,  30, "embedding_3d_front.png"),
              ( 5, -90, "embedding_3d_side.png")]
    for elev, azim, fname in angles:
        print(f"plotting PCA 3D ({fname})...", flush=True)
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2],
                    c="#d0d0d0", s=5, alpha=0.25, linewidths=0)
        for cat, ids in cat_ids.items():
            if not ids:
                continue
            rows = [id_to_row[i] for i in ids if i in id_to_row]
            if not rows:
                continue
            rows = np.array(rows)
            ax.scatter(emb_3d[rows, 0], emb_3d[rows, 1], emb_3d[rows, 2],
                        c=CATEGORY_COLORS[cat], s=55,
                        edgecolors="black", linewidths=0.4,
                        alpha=0.9, label=f"{cat}")
        pct = 100 * var_ratio.sum()
        ax.set_title(f"Evolved embedding — 3D PCA "
                      f"({pct:.1f} % variance explained)",
                      fontsize=12)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        out = os.path.join(assets_dir, fname)
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  saved {out}", flush=True)

    # ----- Chart 5: category centroid cosine heatmap -----
    print("plotting centroid heatmap...", flush=True)
    cats = list(CATEGORIES.keys())
    centroids = []
    for cat in cats:
        ids = cat_ids[cat]
        if not ids:
            centroids.append(np.zeros(emb_np.shape[1], dtype=np.float32))
            continue
        v = emb_np[ids].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        centroids.append(v)
    centroids = np.stack(centroids)
    cosmat = centroids @ centroids.T

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cosmat, vmin=-0.3, vmax=1.0, cmap="RdYlBu_r")
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticklabels(cats)
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, f"{cosmat[i,j]:.2f}",
                     ha="center", va="center",
                     color="black" if abs(cosmat[i,j]) < 0.5 else "white",
                     fontsize=9)
    ax.set_title("Cosine similarity between category-mean embeddings\n"
                  "(diagonal = 1; off-diagonal shows learned semantic structure)",
                  fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(assets_dir, "embedding_centroid_heatmap.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}", flush=True)

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
