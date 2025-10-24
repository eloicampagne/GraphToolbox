from IPython.display import HTML
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import umap

def load_attention_batches(directory_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load and assemble attention matrices saved in batch files from a given directory.

    Each file is expected to be a `.pt` file containing:
    - "attention_weights": list of L tensors, each of shape [E_total, H], 
      where E_total = E_per_graph × num_graphs.
    - "edge_idx": tensor of shape [2, E_per_graph], shared across all graphs.

    The function reconstructs a tensor of shape [L, H, total_num_graphs, E_per_graph]
    by reshaping and stacking the attention data across batches.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing batch files named as "num_batch{i}.pt".

    Returns
    -------
    all_attentions : torch.Tensor of shape [L, H, total_num_graphs, E_per_graph]
        Concatenated attention weights across all batches.
    edge_index : torch.Tensor of shape [2, E_per_graph]
        Edge indices assumed to be consistent across all batches.
    """
    all_graphs = []
    edge_index = None

    batch_files = sorted(
        [f for f in os.listdir(directory_path) if f.startswith("num_batch") and f.endswith(".pt")],
        key=lambda x: int(x.replace("num_batch", "").replace(".pt", ""))
    )

    for file in batch_files:
        path = os.path.join(directory_path, file)
        data = torch.load(path)

        if 'attention_weights' not in data or 'edge_idx' not in data:
            print(f"Fichier {file} mal formé ou incomplet.")
            continue

        attn_per_layer = data['attention_weights']
        edge_index = data['edge_idx'] if edge_index is None else edge_index

        L = len(attn_per_layer)
        E_total, H = attn_per_layer[0].shape
        E_per_graph = edge_index.shape[1]
        num_graphs = E_total // E_per_graph

        attn_tensor = torch.stack(attn_per_layer, dim=0)  # [L, E_total, H]
        attn_tensor = attn_tensor.view(L, num_graphs, E_per_graph, H)  # [L, B, E, H]
        attn_tensor = attn_tensor.permute(0, 3, 1, 2)  # [L, H, B, E]
        all_graphs.append(attn_tensor)

    all_attentions = torch.cat(all_graphs, dim=2)  # [L, H, nb_graph_total, E]
    return all_attentions, edge_index

def compute_attention_statistics(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation of attention scores for each layer and head,
    in a fully vectorized way without looping over individual graphs.

    Parameters
    ----------
    all_attentions : torch.Tensor of shape [L, H, G, E]
        Attention weights for all layers (L), heads (H), graphs (G), and edges (E).
    edge_index : torch.Tensor of shape [2, E]
        Edge index tensor containing sender and receiver node indices.

    Returns
    -------
    avg_attn : torch.Tensor of shape [L, H, N, N]
        Mean attention matrix for each layer and head, aggregated over graphs.
    std_attn : torch.Tensor of shape [L, H, N, N]
        Standard deviation of attention matrices for each layer and head.
    """
    L, H, G, E = all_attentions.shape
    senders = edge_index[0]
    receivers = edge_index[1]
    N = torch.max(edge_index).item() + 1

    mean_vec = all_attentions.mean(dim=2)  # [L, H, E]
    std_vec = all_attentions.std(dim=2)    # [L, H, E]

    avg_attn = torch.zeros((L, H, N, N), dtype=all_attentions.dtype, device=all_attentions.device)
    std_attn = torch.zeros_like(avg_attn)

    for l in range(L):
        for h in range(H):
            avg_attn[l, h].index_put_((senders, receivers), mean_vec[l, h], accumulate=True)
            std_attn[l, h].index_put_((senders, receivers), std_vec[l, h], accumulate=True)

    return avg_attn, std_attn

def plot_attention_statistics(
    avg_attn: torch.Tensor,
    std_attn: torch.Tensor,
    **kwargs
) -> None:
    """
    Plot heatmaps of the average and standard deviation attention matrices
    for each layer and attention head.

    Parameters
    ----------
    avg_attn : torch.Tensor of shape [L, H, N, N]
        Mean attention matrices to be visualized.
    std_attn : torch.Tensor of shape [L, H, N, N]
        Standard deviation matrices to be visualized.

    Returns
    -------
    None
    """
    L, H, N, _ = avg_attn.shape
    figsize    = kwargs.get('figsize', (3.5, 3))
    fontsize   = kwargs.get('fontsize', 14)
    figsize    = (H * figsize[0], L * figsize[1])

    def plot_matrix_grid(data: torch.Tensor, title_prefix: str):
        fig, axes = plt.subplots(L, H, figsize=figsize,
                                 squeeze=False) 
        for l in range(L):
            for h in range(H):
                ax = axes[l][h]
                sns.heatmap(data[l, h].cpu().numpy().T,
                            ax=ax,
                            cmap='rocket_r',
                            cbar=True)
                ax.set_title(f"{title_prefix} - Layer {l}, Head {h}", fontsize=fontsize/1.6)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f"Attention matrices - {title_prefix}", fontsize=fontsize)
        fig.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.show()

    plot_matrix_grid(avg_attn, "Average")
    plot_matrix_grid(std_attn, "Std")

def animate_grouped_attention(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor,
    group_variable: list | np.ndarray,
    group_name: str = "Group",
    interval: int = 1000,
    mode: str = "mean",
    save: bool = True,
    **kwargs
):
    """
    Animate attention matrices (mean or standard deviation) across groups of graphs.

    Parameters
    ----------
    all_attentions : torch.Tensor of shape [L, H, G, E]
        Attention scores for all layers (L), heads (H), graphs (G), and edges (E).
    edge_index : torch.Tensor of shape [2, E]
        Edge indices indicating source and target nodes.
    group_variable : array-like of length G
        Group identifier for each graph (e.g., time step, class, or cluster ID).
    group_name : str, optional (default="Group")
        Name of the group variable to display in the animation title.
    interval : int, optional (default=1000)
        Time interval between frames in milliseconds.
    mode : {"mean", "std"}, optional (default="mean")
        Statistic to visualize: either the mean or standard deviation of attention scores.
    save : bool, optional (default=True)
        Save the figure to .gif.
    Returns
    -------
    IPython.display.HTML
        HTML animation of attention matrices for each group, rendered in Jupyter notebooks.
    """
    assert mode in ("mean", "std"), "mode must be 'mean' or 'std'"
    L, H, G, E = all_attentions.shape
    base_figsize = kwargs.get('figsize', (3.5, 3))
    vmin = kwargs.get('vmin')
    vmax = kwargs.get('vmax')
    figsize = (H * base_figsize[0], L * base_figsize[1])
    gif_path = kwargs.get("gif_path", os.path.join('./attention_matrix', f'attention_animation_{mode}.gif'))

    group_variable = np.array(group_variable[:G])
    groups = np.unique(group_variable)

    senders, receivers = edge_index[0].long(), edge_index[1].long()
    N = int(max(senders.max(), receivers.max()).item()) + 1

    matrices_per_group = []
    for group_val in groups:
        mask = (group_variable == group_val)
        indices = torch.tensor(np.where(mask)[0], dtype=torch.long, device=all_attentions.device)
        attn_group = all_attentions[:, :, indices, :]  # [L, H, Gg, E]
        vec = attn_group.mean(dim=2) if mode == "mean" else attn_group.std(dim=2)

        mat = torch.zeros((L, H, N, N), dtype=vec.dtype)
        for l in range(L):
            for h in range(H):
                mat[l, h].index_put_((senders, receivers), vec[l, h], accumulate=True)
        matrices_per_group.append(mat.cpu().numpy())

    fig, axes = plt.subplots(L, H, figsize=figsize, squeeze=False)
    ims = []
    for l in range(L):
        row = []
        for h in range(H):
            ax = axes[l][h]
            im = ax.imshow(matrices_per_group[0][l][h], cmap='rocket_r', vmin=vmin, vmax=vmax)
            ax.set_title(f"Layer {l} - Head {h}", fontsize=10)
            ax.set_xlabel("Target")
            ax.set_ylabel("Source")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row.append(im)
        ims.append(row)

    def update(i):
        fig.suptitle(f"{mode.capitalize()} Attention - {group_name} = {groups[i]}", fontsize=14)
        for l in range(L):
            for h in range(H):
                ims[l][h].set_data(matrices_per_group[i][l][h])
        return sum(ims, [])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    anim = FuncAnimation(fig, update, frames=len(groups), interval=interval, blit=False)
    
    if save:
        anim.save(gif_path, writer='pillow', dpi=200)
        print(f"GIF saved at: {os.path.abspath(gif_path)}")

    plt.close(fig)
    return HTML(anim.to_jshtml())

def pca_analysis_attention(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    n_components: int = 10
) -> None:
    """
    Perform Principal Component Analysis (PCA) on a selected attention head across graphs.

    This function visualizes:
    - The explained variance of the principal components (PCs),
    - A 2D projection of the data on the first two PCs,
    - Heatmaps of the top principal components reshaped into attention matrices.

    Parameters
    ----------
    all_attentions : torch.Tensor of shape [L, H, G, E] or [G, E]
        Attention values. Can be the full tensor from a model or already flattened for a given head.
    edge_index : torch.Tensor of shape [2, E]
        Edge indices (source and target nodes).
    layer_idx : int, optional (default=0)
        Index of the attention layer to analyze.
    head_idx : int, optional (default=0)
        Index of the attention head to analyze.
    n_components : int, optional (default=10)
        Number of principal components to extract and visualize.

    Returns
    -------
    None
        Displays plots directly.
    """
    if len(all_attentions.shape) == 4:
        L, H, G, E = all_attentions.shape
        src, tgt = edge_index
        att_flat = all_attentions[layer_idx, head_idx, :, :]  # [G, E]
        att_flat = att_flat.cpu().numpy()
        N = int(src.max().item() + 1)
    elif len(all_attentions.shape) == 2: 
        att_flat = all_attentions
        N = int(all_attentions.shape[1]**0.5)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(att_flat)
    explained_variance = pca.explained_variance_ratio_
    components = pca.components_  # [n_components, E]

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_components + 1), explained_variance, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance")
    plt.title("Energy of each principal component")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    time_labels = np.linspace(0, 1, X_pca.shape[0])
    hsv_colors = time_labels 

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hsv_colors, cmap='hsv', s=1)
    plt.title(f"PCA of Attention vectors\n(Layer {layer_idx}, Head {head_idx})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    cbar = plt.colorbar(scatter, label="DoY")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['1 Jan', '1 Apr', '1 Jul', '1 Oct', '31 Dec'])
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    if n_components == 1:
        axes = [axes]

    for idx, comp in enumerate(components):
        mat = torch.zeros(N, N)
        if len(all_attentions.shape) == 4:
            mat[src, tgt] = torch.tensor(comp)
            mat = mat.numpy()
        elif len(all_attentions.shape) == 2: 
            mat = comp.reshape(N,N)

        sns.heatmap(
            mat, 
            cmap="rocket_r", 
            ax=axes[idx], 
            vmin=-1, vmax=1, 
            cbar=True
        )
        axes[idx].set_title(f"PC{idx+1}")

    plt.suptitle(f"PCA of Attention matrices (Layer {layer_idx}, Head {head_idx})", fontsize=16)
    plt.tight_layout()
    plt.show()

def umap_analysis_attention(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2
) -> None:
    """
    Perform UMAP-based dimensionality reduction on a selected attention head across graphs.

    This function visualizes:
    - A low-dimensional embedding of attention vectors using UMAP,
    - A 2D scatter plot colored cyclically to reflect graph ordering (e.g., temporal).

    Parameters
    ----------
    all_attentions : torch.Tensor of shape [L, H, G, E] or [G, E]
        Attention weights either for the entire model or already extracted for a given head.
    edge_index : torch.Tensor of shape [2, E]
        Edge indices (source and target nodes), required to infer node count if needed.
    layer_idx : int, optional (default=0)
        Index of the attention layer to analyze.
    head_idx : int, optional (default=0)
        Index of the attention head to analyze.
    n_neighbors : int, optional (default=15)
        Number of neighbors for the UMAP algorithm (controls local/global structure).
    min_dist : float, optional (default=0.1)
        Minimum distance between embedded points (controls tightness of clusters).
    n_components : int, optional (default=2)
        Number of output dimensions (typically 2 for visualization).

    Returns
    -------
    None
        Displays a UMAP projection plot.
    """
    if len(all_attentions.shape) == 4:
        L, H, G, E = all_attentions.shape
        src, tgt = edge_index
        att_flat = all_attentions[layer_idx, head_idx, :, :]  # [G, E]
    elif len(all_attentions.shape) == 2:
        att_flat = all_attentions

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean'
    )
    X_umap = reducer.fit_transform(att_flat)  # [G, n_components]

    time_labels = np.linspace(0, 1, X_umap.shape[0])
    hsv_colors = time_labels  
    plt.figure(figsize=(6, 6))
    if n_components == 2:
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=hsv_colors, cmap='hsv', s=1)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"UMAP Projection (Layer {layer_idx}, Head {head_idx})")
        cbar = plt.colorbar(scatter, label="DoY")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(['1 Jan', '1 Apr', '1 Jul', '1 Oct', '31 Dec'])
    else:
        plt.plot(X_umap)
        plt.title(f"UMAP Projection on {n_components} dimensions")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def normalized_laplacian(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute the symmetric normalized Laplacian of an adjacency matrix.

    Parameters
    ----------
    A : torch.Tensor of shape [N, N]
        Adjacency matrix of the graph (must be square and 2D).
    eps : float, optional
        Small value added to avoid division by zero.

    Returns
    -------
    L : torch.Tensor of shape [N, N]
        Symmetric normalized Laplacian matrix: L = I - D^{-1/2} A D^{-1/2}.
    """
    if A.ndim != 2:
        raise ValueError(f"Expected 2D square matrix, got shape {A.shape}")
    deg = torch.sum(A, dim=1)
    deg_sqrt_inv = torch.zeros_like(deg)
    nonzero = deg > 0
    deg_sqrt_inv[nonzero] = 1.0 / torch.sqrt(deg[nonzero] + eps)
    D_inv_sqrt = torch.diag(deg_sqrt_inv)
    L = torch.eye(A.shape[0], device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def plot_spectral_gap(L: torch.Tensor, max_k: int) -> tuple[np.ndarray, int]:
    """
    Plot the first `max_k` eigenvalues of a Laplacian and estimate the optimal number of clusters via the spectral gap.

    Parameters
    ----------
    L : torch.Tensor of shape [N, N]
        Laplacian matrix.
    max_k : int
        Number of smallest eigenvalues to consider.

    Returns
    -------
    eigvals : np.ndarray
        First `max_k` eigenvalues of the Laplacian.
    optimal_k : int
        Estimated number of clusters based on the largest spectral gap (elbow method).
    """
    eigvals,_ = torch.linalg.eigh(L)
    eigvals = eigvals[:max_k].numpy()
    gaps = eigvals[1:] - eigvals[:-1]
    optimal_k = int(gaps.argsort()[-2]) + 1
    plt.figure(figsize=(6, 4))
    plt.plot(range(0, max_k), eigvals, marker='o', label="Valeurs propres")
    plt.axvline(optimal_k, color='red', linestyle='--', label=f"Coude (k={optimal_k})")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Elbow Method")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return eigvals, optimal_k

def spectral_embedding(L: torch.Tensor, k: int, plot: bool = False) -> torch.Tensor:
    """
    Compute the spectral embedding (first `k` eigenvectors of the Laplacian).

    Parameters
    ----------
    L : torch.Tensor of shape [N, N]
        Laplacian matrix.
    k : int
        Number of leading eigenvectors to return.
    plot : bool, optional
        If True, plots the spectrum of eigenvalues.

    Returns
    -------
    embedding : torch.Tensor of shape [N, k]
        Matrix of the first `k` eigenvectors.
    """
    eigvals, eigvecs = torch.linalg.eigh(L)
    if plot:
        plt.plot(eigvals)
        plt.show()
    return eigvecs[:, :k]  # (N, k)

def cosine_similarity_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity matrix between rows of a matrix.

    Parameters
    ----------
    X : torch.Tensor of shape [N, d]
        Input feature matrix.

    Returns
    -------
    similarity : torch.Tensor of shape [N, N]
        Cosine similarity between all pairs of rows in X.
    """
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-6)
    return X_norm @ X_norm.T

def spectral_fusion(lA: list[torch.Tensor], k: int, **kwargs) -> torch.Tensor:
    """
    Perform spectral fusion by computing the average cosine similarity 
    of the spectral embeddings of several adjacency matrices.

    Parameters
    ----------
    lA : list of torch.Tensor [N, N]
        List of adjacency matrices to fuse.
    k : int
        Number of eigenvectors to use for each spectral embedding.
    **kwargs : dict
        Optional arguments passed to `spectral_embedding`.

    Returns
    -------
    A_fused : torch.Tensor of shape [N, N]
        Fused similarity matrix.
    """
    lE = []
    for A in lA:
        L = normalized_laplacian(A)
        E = spectral_embedding(L, k, **kwargs)
        lE.append(E)
    max_dim = max(E.shape[1] for E in lE)
    lE_padded = [torch.nn.functional.pad(E, (0, max_dim - E.shape[1])) for E in lE]
    E_fused = torch.mean(torch.stack(lE_padded), axis=0)
    A_fused = cosine_similarity_matrix(E_fused)
    return A_fused

def hierarchical_attention_fusion(attn_tensor: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
    """
    Fuse attention maps hierarchically across heads and layers using spectral fusion.

    Parameters
    ----------
    attn_tensor : torch.Tensor of shape [L, H, N, N]
        Attention matrices for L layers and H heads.
    k : int
        Number of eigenvectors used in spectral embeddings.
    **kwargs : dict
        Optional arguments passed to `spectral_fusion`.

    Returns
    -------
    A_final : torch.Tensor of shape [N, N]
        Final fused similarity matrix after hierarchical fusion.
    """
    L, H, N, _ = attn_tensor.shape
    fused_per_layer = []
    for l in range(L):
        heads = [attn_tensor[l, h] for h in range(H)]
        fused_layer = spectral_fusion(heads, k, **kwargs)
        fused_per_layer.append(fused_layer)
    A_final = spectral_fusion(fused_per_layer, k, **kwargs)
    return A_final