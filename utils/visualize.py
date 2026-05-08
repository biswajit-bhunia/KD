import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_embeddings(embeddings, labels, save_path="tsne.png"):
    """
    embeddings: (N, D)
    labels: (N,)
    """
    embeddings = embeddings.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(6, 6))

    for label in set(labels):
        idx = labels == label
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=f"class {label}", alpha=0.6)

    plt.legend()
    plt.title("t-SNE Embedding")
    plt.grid()

    plt.savefig(save_path)
    plt.close()