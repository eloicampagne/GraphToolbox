import torch
from torch_geometric.nn import GATv2Conv, TransformerConv, GCNConv, SAGEConv
from torch.nn import Linear, ReLU, LayerNorm
from inspect import signature
from einops import rearrange
from torch_geometric.nn.models.deepgcn import DeepGCNLayer

import torch
from torch_geometric.nn import GATv2Conv, TransformerConv, GCNConv, SAGEConv
from torch.nn import Linear, ReLU, LayerNorm
from inspect import signature
from einops import rearrange
from torch_geometric.nn.models.deepgcn import DeepGCNLayer

class myGNN(torch.nn.Module):
    """
    A flexible deep Graph Neural Network supporting various convolution types
    (GATv2, TransformerConv, GCN, GraphSAGE) with residual connections.

    This model stacks multiple message-passing layers (`DeepGCNLayer`) and supports
    optional attention-weight extraction for interpretability.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    num_layers : int
        Number of GNN layers to stack.
    hidden_channels : int
        Hidden dimension per layer.
    out_channels : int
        Output dimension (e.g. number of regression or classification targets).
    conv_class : torch_geometric.nn.conv.MessagePassing, optional
        Convolution class to use (default: ``GATv2Conv``).
    conv_kwargs : dict, optional
        Additional keyword arguments for the convolution layer (e.g. heads, dropout).

    Attributes
    ----------
    node_encoder : torch.nn.Linear
        Linear layer mapping input features to hidden space.
    layers : torch.nn.ModuleList
        Sequence of `DeepGCNLayer` blocks with residual connections.
    norm_final : torch.nn.LayerNorm
        Final normalization layer.
    fc : torch.nn.Linear
        Output linear layer mapping hidden features to predictions.

    Notes
    -----
    - When using multi-head attention convolutions (e.g. GATv2), the model automatically
      adjusts internal dimensions.
    - Attention weights can be returned via `return_attention=True` for visualization.

    Examples
    --------
    >>> model = myGNN(in_channels=16, hidden_channels=32, out_channels=1, num_layers=3)
    >>> out = model(x, edge_index)
    >>> out.shape
    torch.Size([N, 1])
    """
    def __init__(self,
                 in_channels,
                 num_layers,
                 hidden_channels,
                 out_channels,
                 **kwargs):
        super(myGNN, self).__init__()

        self.conv_class = kwargs.get('conv_class', GATv2Conv)
        self.conv_kwargs = kwargs.get('conv_kwargs', {})

        conv_params = signature(self.conv_class).parameters
        self.use_heads = 'heads' in conv_params
        self.heads = self.conv_kwargs.get('heads', 1) if self.use_heads else 1

        in_dim = hidden_channels * self.heads if self.use_heads else hidden_channels
        out_dim = hidden_channels

        self.in_channels = in_channels
        self.node_encoder = Linear(in_channels, in_dim)
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            valid_keys = conv_params.keys()
            conv_args = {}
            potential_args = {
                'in_channels': in_dim,
                'out_channels': out_dim,
                'heads': self.heads,
                'concat': True,
                'edge_dim': 1,
                'add_self_loops': False
            }

            for k, v in self.conv_kwargs.items():
                if k in valid_keys:
                    conv_args[k] = v

            for k, v in potential_args.items():
                if k in valid_keys and k not in conv_args:
                    conv_args[k] = v

            conv = self.conv_class(**conv_args)
            norm = LayerNorm(in_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv=conv, norm=norm, act=act, block='res+', ckpt_grad=True)
            self.layers.append(layer)

        self.norm_final = LayerNorm(in_dim, elementwise_affine=True)
        self.fc = Linear(in_dim, out_channels)

    def forward(self, x, edge_index, edge_weight=None, return_attention=False, **kwargs):
        """
        Forward pass through the graph neural network.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[N, F]`` or ``[N, B, F]``.
        edge_index : torch.LongTensor
            Graph connectivity in COO format.
        edge_weight : torch.Tensor, optional
            Edge weights, shape ``[E]`` or ``[E, 1]`` depending on the convolution.
        return_attention : bool, default=False
            If True, also returns layer-wise attention weights.
        **kwargs :
            Optional keys when ``return_attention=True``:
                - ``batch_size`` : int
                - ``num_edges_per_graph`` : int
                - ``save`` : bool
                - ``save_path`` : str

        Returns
        -------
        torch.Tensor or tuple
            Model output (and optionally attention weights dictionary).
        """
        if x.dim() == 3:
            # garder N,B pour construire l’edge_index bloc-diagonal si besoin
            N_per_graph, B, _ = x.shape
            x = rearrange(x, 'n b c -> (b n) c')
        else:
            N_per_graph, B = x.shape[0], 1

        if edge_weight is not None and self.conv_class == TransformerConv:
            edge_weight = edge_weight.unsqueeze(1)
        x = self.node_encoder(x)

        # Préparer edge_index/edge_weight locaux (bloc-diagonal si B>1)
        local_edge_index = edge_index
        local_edge_weight = edge_weight
        if return_attention and B > 1:
            local_edge_index = _make_block_diag_edge_index(edge_index, N_per_graph, B)
            if edge_weight is not None:
                # on répète les poids sur chaque graphe
                repeat_times = B if edge_weight.dim() == 1 else (B, 1)
                local_edge_weight = edge_weight.repeat(repeat_times)

        if return_attention:
            save = kwargs.get("save", False)
            save_path = kwargs.get("save_path", None)
            num_edges_per_graph = edge_index.size(1)
            num_graphs = B
            if save:
                attentions = {
                    "attention_weights": [],   # liste (L) de tensors [E_graph*B, H]
                    "edge_idx": edge_index.detach().cpu(),  # [2, E_graph] du graphe simple
                    "num_graphs": num_graphs,  # B
                }
            else:
                attentions = {
                    "first_graph": [],         # par couche: (weights_first_graph [E_graph,H], edge_idx_first [2,E_graph])
                    "mean": [],                # par couche: [E_graph, H] moyenne sur B
                    "std": []                  # par couche: [E_graph, H] std sur B
                }

        for _, layer in enumerate(self.layers):
            if return_attention:
                _, attn = layer.conv(x, local_edge_index, local_edge_weight, return_attention_weights=True)
                edge_idx_all, attn_weights_all = attn  # edge_idx_all: [2, E_graph*B]; attn_weights_all: [E_graph*B, H]

                if save:
                    attentions["attention_weights"].append(attn_weights_all.detach().cpu())
                    del attn_weights_all
                else:
                    # reshape propre: [B, E_graph, H]
                    att_reshaped = attn_weights_all.view(num_graphs, num_edges_per_graph, self.heads)
                    att_first = att_reshaped[0]  # [E_graph, H]
                    attentions["first_graph"].append((att_first.cpu().detach(), edge_index.cpu().detach()))
                    attentions["mean"].append(att_reshaped.mean(dim=0).cpu().detach())
                    attentions["std"].append(att_reshaped.std(dim=0).cpu().detach())
                    del att_reshaped, attn_weights_all

            if 'edge_weight' in signature(layer.conv.forward).parameters or 'edge_attr' in signature(layer.conv.forward).parameters:
                x = layer(x, local_edge_index, local_edge_weight)
            else:
                x = layer(x, local_edge_index)

        x = self.layers[0].act(self.norm_final(x))
        x = self.fc(x)

        if return_attention and save:
            if save_path is None:
                raise ValueError("You must provide `save_path` when `save=True`.")
            torch.save(attentions, save_path)

        return (x, attentions) if return_attention else x

def _make_block_diag_edge_index(edge_index: torch.Tensor, num_nodes: int, batch_size: int) -> torch.Tensor:
    """
    Create a block-diagonal edge_index representing a batch of disjoint, identical graphs.
    Parameters
    ----------
    edge_index : torch.Tensor
        2 x E tensor of edge indices (source, target) for a single graph.
    num_nodes : int
        Number of nodes in the single graph (used to offset node indices).
    batch_size : int
        Number of graph copies to stack; if 1 the original edge_index is returned.
    Returns
    -------
    torch.Tensor
        2 x (batch_size * E) edge_index for the disjoint union of the batch.
        The returned tensor has the same dtype and device as the input.
    Notes
    -----
    Each copy's node indices are shifted by k * num_nodes (for k in [0, batch_size-1])
    so that graphs are kept disjoint when combined.
    """
    
    if batch_size == 1:
        return edge_index
    offsets = torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype) * num_nodes
    src = edge_index[0].unsqueeze(1) + offsets.unsqueeze(0)  # [E, B]
    dst = edge_index[1].unsqueeze(1) + offsets.unsqueeze(0)  # [E, B]
    eidx_bd = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # [2, B*E]
    return eidx_bd

class GCNEncoder(torch.nn.Module):
    """
    Simple 2-layer Graph Convolutional Network encoder.

    This encoder maps node features into a compact latent space using
    two GCNConv layers and ReLU activation.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    out_channels : int
        Dimension of the latent embedding space.

    Examples
    --------
    >>> enc = GCNEncoder(in_channels=32, out_channels=16)
    >>> z = enc(x, edge_index)
    >>> z.shape
    torch.Size([N, 16])
    """
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels) 
        self.conv2 = GCNConv(2 * out_channels, out_channels) 

    def forward(self, x, edge_index):
        """
        Forward pass through the GCN encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[N, F]``.
        edge_index : torch.LongTensor
            Graph connectivity in COO format.

        Returns
        -------
        torch.Tensor
            Latent node representations of shape ``[N, out_channels]``.
        """
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGNNEncoder(torch.nn.Module):
    """
    Variational Graph Encoder producing mean and log-variance embeddings
    for Variational Graph Autoencoders (VGAE) or graph-based latent models.

    Supports both GCN and GraphSAGE convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    out_channels : int
        Latent embedding dimension.
    conv : {'gcn', 'sage'}, default='gcn'
        Type of graph convolution to use.

    Attributes
    ----------
    conv1 : torch_geometric.nn.MessagePassing
        First convolution layer (shared for both mu/logstd branches).
    conv_mu : torch_geometric.nn.MessagePassing
        Convolution layer producing mean embeddings.
    conv_logstd : torch_geometric.nn.MessagePassing
        Convolution layer producing log-variance embeddings.

    Examples
    --------
    >>> enc = VariationalGNNEncoder(in_channels=32, out_channels=16, conv='sage')
    >>> mu, logstd = enc(x, edge_index)
    >>> mu.shape, logstd.shape
    (torch.Size([N, 16]), torch.Size([N, 16]))
    """
    def __init__(self, in_channels, out_channels, conv='gcn'):
        super(VariationalGNNEncoder, self).__init__()
        if conv == 'gcn':
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
            self.conv_mu = GCNConv(2 * out_channels, out_channels)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        else:
            self.conv1 = SAGEConv(in_channels, 2 * out_channels)
            self.conv_mu = SAGEConv(2 * out_channels, out_channels)
            self.conv_logstd = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Compute latent mean and log-variance representations.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[N, F]``.
        edge_index : torch.LongTensor
            Graph connectivity in COO format.

        Returns
        -------
        tuple of torch.Tensor
            Mean and log-variance tensors, each of shape ``[N, out_channels]``.
        """
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)