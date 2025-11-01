from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
import networkx as nx
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Optional, Any, Dict, List, Tuple

from graphtoolbox.data.dataset import GraphDataset

@dataclass
class VisualizationConfig:
    name: str = "default"                       
    output_root: str = "interpretability"       

    # Map and positions
    basemap: Optional[Dict[str, Any]] = None    
    positions: Optional[Dict[int, Tuple[float, float]]] = None  # {node_id: (lon, lat)}
    pos_df: Optional[pd.DataFrame] = None
    lon_col: str = "LONGITUDE"
    lat_col: str = "LATITUDE"
    node_id_col: Optional[str] = None

    # Time grouping config
    grouping: Dict[str, Any] = field(default_factory=lambda: dict(mode="all", ndays=3, labels=None, indices=None))
    start_date_key: str = "day_inf_test"        # key in data_kwargs to reconstruct dates
    date_freq: str = "D"                        # pandas frequency string (e.g. 'D', 'H')

    # Basemap defaults (used if not provided in basemap)
    map_projection: str = "merc"
    map_resolution: str = "i"
    draw_coastlines: bool = True
    draw_countries: bool = True
    fillcontinents_color: str = "gray"
    mapboundary_color: str = "white"

    # Drawing options
    show_nodes: bool = True
    show_labels: bool = True
    node_size: int = 500
    node_color: str = "blue"
    node_alpha: float = 0.6
    label_fontsize: int = 10
    label_color: str = "white"

    edge_cmap: str = "rocket_r"                 # colormap when vis_mode != "std"
    edge_cmap_std: str = "mako"                 # colormap when vis_mode == "std"
    edge_width_min: float = 1.0
    edge_width_max: float = 5.0
    edge_alpha: float = 1.0
    connectionstyle: str = "arc3,rad=0.1"
    
    # Edge arrows (use FancyArrowPatches). Warning: can be slow on large graphs.
    edge_arrows: bool = False
    arrowstyle: str = "-|>"
    arrowsize: int = 10
    arrow_max_edges: int = 500

    # Edge importance weighting when vis_mode == "context"
    normalize_with_edge_weight: bool = True

    # Output options
    save_dpi: int = 150
    file_ext: str = "pdf"
    subdir: str = "explanation_graph"

    # Font sizes
    fontsize: int = 16
    labelsize: int = 12

# ----------------------------
# --- Internal helper utils ---
# ----------------------------

def _month_indices(month_name, dates):
    """Return the temporal indices corresponding to a given month."""
    if month_name == "ALL" or dates is None:
        return slice(None)
    return np.where(dates.month_name() == month_name)[0]


def _pick_temporal_mask(exp_dict, vis_mode, idxs):
    """Select and aggregate a temporal edge mask according to vis_mode."""
    em_time = exp_dict.get("edge_masks", None)
    if em_time is None:
        em_time = globals().get("edge_masks", None)
    if em_time is not None and not isinstance(idxs, slice):
        if vis_mode in ["model", "context"]:
            return em_time[idxs].mean(dim=0).detach().cpu().numpy()
        else:
            return em_time[idxs].std(dim=0).detach().cpu().numpy()
    # fallback: global or per-explanation
    if vis_mode in ["model", "context"]:
        return exp_dict["mean"].edge_mask.detach().cpu().numpy()
    else:
        return exp_dict["std"].detach().cpu().numpy()


# ----------------------------
# --- Main visualization ---
# ----------------------------

# Fallback positions: if no geographic positions were set, assign deterministic pseudo-random lon/lat within map extent.
    
def _fill_missing_positions(pos_map: Dict[int, Tuple[float, float]],
                            n_nodes: int,
                            map_kwargs: Dict[str, Any],
                            seed: int = 42) -> Dict[int, Tuple[float, float]]:
    """
    Fill missing positions by sampling them uniformly within the Basemap extent.
    Sampling is deterministic and controlled by 'seed'.
    """
    have = set(int(k) for k in pos_map.keys())
    missing = [i for i in range(n_nodes) if i not in have]
    if not missing:
        return pos_map
    lon_lo = float(map_kwargs.get('llcrnrlon', -5))
    lon_hi = float(map_kwargs.get('urcrnrlon', 10))
    lat_lo = float(map_kwargs.get('llcrnrlat', 40))
    lat_hi = float(map_kwargs.get('urcrnrlat', 52))
    rng = np.random.RandomState(seed)
    updated = dict(pos_map)
    for i in missing:
        lon_i = float(rng.uniform(lon_lo, lon_hi))
        lat_i = float(rng.uniform(lat_lo, lat_hi))
        updated[int(i)] = (lon_i, lat_i)
    return updated

def plot_explanation_graph(
    all_explanations: dict,
    graph_dataset_test: GraphDataset,
    data_kwargs: dict,
    dataset: str = "default",                      # kept for backward compatibility: only used as default cfg.name
    vis_mode: str = "std",
    months_to_plot: list[str] = ["ALL"],           # backward compat; overridden by cfg.grouping if provided
    edge_keep_ratio: float = 0.10,
    df_pos: pd.DataFrame | None = None,         # backward compat; prefer cfg.positions or cfg.pos_df
    viz_cfg: VisualizationConfig | dict | None = None
):
    """
    Visualize explanation graphs on a map for the selected period.

    Parameters
    ----------
    all_explanations : dict[str, dict]
        Dictionary {model_name: {"mean": expl, "std": expl, "edge_masks": ...}}
    graph_dataset_test : list[Data]
        Test set containing PyG graphs (at least one element)
    data_kwargs : dict
        Must contain the key 'day_inf_test' (start date of the test set)
    dataset : str
        Kept for backward compatibility; if viz_cfg is provided, its name is used instead.
    vis_mode : str
        "model", "context", or "std"
    months_to_plot : list[str]
        Backward compat. If viz_cfg.grouping is provided, it takes precedence.
    edge_keep_ratio : float
        Percentage of edges to keep (0.10 = top 10%)
    df_pos : pd.DataFrame, optional
        Backward compat. Prefer viz_cfg.positions or viz_cfg.pos_df.
    viz_cfg : VisualizationConfig | dict, optional
        User configuration providing positions, map kwargs, grouping, etc.
    """
    # --- Build configuration ---
    if viz_cfg is None:
        cfg = VisualizationConfig(name=str(dataset))
        # Legacy: build positions from df_villes if provided
        if df_pos is not None and all(c in df_pos.columns for c in ["LATITUDE", "LONGITUDE"]):
            cfg.pos_df = df_pos
            cfg.lon_col, cfg.lat_col = "LONGITUDE", "LATITUDE"
    elif isinstance(viz_cfg, dict):
        cfg = VisualizationConfig(**viz_cfg)
    else:
        cfg = viz_cfg

    # --- Reconstruct dates from temporal edge_masks ---
    dates = None
    try:
        t_len = None
        for _m, _exp in all_explanations.items():
            if _exp.get("edge_masks", None) is not None:
                t_len = int(_exp["edge_masks"].shape[0])
                break
        if t_len is None:
            em_glob = globals().get("edge_masks", None)
            if em_glob is not None:
                try:
                    t_len = int(em_glob.shape[0])
                except Exception:
                    t_len = None
        if t_len is not None:
            start = pd.to_datetime(data_kwargs["day_inf_test"])
            dates = pd.date_range(start=start, periods=t_len, freq="D")
    except Exception:
        dates = None

    # --- Loop over models ---
    for model_name, exp_dict in all_explanations.items():
        explanation = exp_dict["mean"]
        edge_index = explanation.edge_index.cpu().numpy()

        G_base = nx.Graph()
        pos_map = _build_positions(cfg, graph_dataset_test)
        for nid, (lon, lat) in pos_map.items():
            G_base.add_node(int(nid), pos=(float(lon), float(lat)))

        pos_all = nx.get_node_attributes(G_base, 'pos')
        map_kwargs = _infer_map_extent(pos_all, cfg)

        data_graph = graph_dataset_test[0]
        n_nodes = _infer_num_nodes(data_graph, explanation, edge_index)

        pos_map = _fill_missing_positions(pos_map, n_nodes, map_kwargs)
        G_base.clear()
        for nid, (lon, lat) in pos_map.items():
            G_base.add_node(int(nid), pos=(float(lon), float(lat)))

        # -----------------------------
        # 4. Normalize edge_weight if present
        # -----------------------------
        base_ei = data_graph.edge_index
        base_ew = getattr(data_graph, 'edge_weight', None)
        ew_map, ew_min, ew_max, ew_den = _prepare_edge_weight_map(base_ei, base_ew)

        # -----------------------------
        # 5. Determine temporal panels (groups)
        # -----------------------------
        panel_labels, panel_indices, nrows, ncols, tag = _define_panels(cfg, dates, months_to_plot)

        # -----------------------------
        # 6. Prepare data to plot
        # -----------------------------
        vmin_global, vmax_global = float('inf'), float('-inf')
        panel_data = []
        for lbl, idxs in zip(panel_labels, panel_indices):
            edge_mask = _pick_temporal_mask(exp_dict, vis_mode, idxs)
            edges_draw, weights_draw = _select_top_edges(
                edge_index, edge_mask, vis_mode,
                base_ew if cfg.normalize_with_edge_weight else None,
                ew_map, ew_min, ew_den, edge_keep_ratio
            )
            if weights_draw:
                vmin_global = min(vmin_global, min(weights_draw))
                vmax_global = max(vmax_global, max(weights_draw))
            panel_data.append((lbl, edges_draw, weights_draw))

        if not np.isfinite(vmin_global):
            vmin_global, vmax_global = 0.0, 1.0

        # -----------------------------
        # 7. Draw the graphs on the map
        # -----------------------------
        _draw_graph_panels(
            model_name, cfg.name, cfg, vis_mode,
            panel_data, G_base, map_kwargs,
            vmin_global, vmax_global, nrows, ncols, tag,
            output_root=cfg.output_root
        )


# ----------------------------
# --- Helper functions ---
# ----------------------------

def _build_positions(cfg: VisualizationConfig, graph_dataset_test: list) -> Dict[int, Tuple[float, float]]:
    """
    Build node positions (lon, lat) from cfg or graph.
    Priority: cfg.positions -> cfg.pos_df -> graph attributes ('pos'/'coords' or ('lon','lat')).
    """
    # 1) Explicit mapping
    if cfg.positions is not None:
        return {int(k): (float(v[0]), float(v[1])) for k, v in cfg.positions.items()}

    # 2) DataFrame with lon/lat columns
    if cfg.pos_df is not None and cfg.lon_col in cfg.pos_df.columns and cfg.lat_col in cfg.pos_df.columns:
        if cfg.node_id_col and cfg.node_id_col in cfg.pos_df.columns:
            ids = cfg.pos_df[cfg.node_id_col].astype(int).tolist()
        else:
            ids = list(range(len(cfg.pos_df)))
        lons = cfg.pos_df[cfg.lon_col].astype(float).tolist()
        lats = cfg.pos_df[cfg.lat_col].astype(float).tolist()
        return {int(i): (float(lo), float(la)) for i, lo, la in zip(ids, lons, lats)}

    # 3) Read from graph attributes
    try:
        d = graph_dataset_test[0]
        coords = None
        for attr in ['pos', 'coords']:
            if hasattr(d, attr) and getattr(d, attr) is not None:
                coords = getattr(d, attr).detach().cpu().numpy()
                break
        if coords is None and hasattr(d, 'lon') and hasattr(d, 'lat'):
            coords = np.stack([d.lon.detach().cpu().numpy(), d.lat.detach().cpu().numpy()], axis=1)
        if coords is not None:
            return {int(i): (float(lon), float(lat)) for i, (lon, lat) in enumerate(coords)}
    except Exception:
        pass
    return {}


def _infer_map_extent(pos_all: Dict[int, Tuple[float, float]], cfg: VisualizationConfig):
    """Return projection parameters suited to provided positions or cfg.basemap."""
    # If user provided explicit Basemap kwargs, use them
    if cfg.basemap is not None:
        return cfg.basemap
    # auto extent from positions
    if len(pos_all) > 0:
        lons = np.array([p[0] for p in pos_all.values()], dtype=float)
        lats = np.array([p[1] for p in pos_all.values()], dtype=float)
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        pad_lon = max(0.2, 0.1 * (lon_max - lon_min + 1e-12))
        pad_lat = max(0.2, 0.1 * (lat_max - lat_min + 1e-12))
        return dict(
            projection=cfg.map_projection,
            llcrnrlon=lon_min - pad_lon, urcrnrlon=lon_max + pad_lon,
            llcrnrlat=lat_min - pad_lat, urcrnrlat=lat_max + pad_lat
        )
    # fallback
    return dict(projection=cfg.map_projection, llcrnrlon=-5, urcrnrlon=10, llcrnrlat=40, urcrnrlat=52)

def _infer_num_nodes(data_graph, explanation, edge_index):
    """Try to determine the number of nodes in a graph."""
    try:
        if hasattr(data_graph, 'num_nodes') and data_graph.num_nodes is not None:
            return int(data_graph.num_nodes)
        if hasattr(explanation, 'x') and explanation.x is not None:
            return int(explanation.x.shape[0])
    except Exception:
        pass
    return int(edge_index.max() + 1)


def _prepare_edge_weight_map(base_ei, base_ew):
    """Build a dictionary {(a,b): weight} and return min/max/denominator."""
    ew_map = {}
    if base_ew is None:
        return ew_map, 0.0, 0.0, 1.0
    ei_np = base_ei.cpu().numpy()
    ew_np = base_ew.detach().cpu().numpy()
    for k in range(ei_np.shape[1]):
        a, b = int(ei_np[0, k]), int(ei_np[1, k])
        w = float(ew_np[k])
        ew_map[(a, b)] = w
        ew_map[(b, a)] = w
    ew_vals = np.array(list(ew_map.values()))
    ew_min, ew_max = ew_vals.min(), ew_vals.max()
    ew_den = (ew_max - ew_min) if ew_max > ew_min else 1.0
    return ew_map, ew_min, ew_max, ew_den


def _define_panels(cfg: VisualizationConfig, dates, months_to_plot_legacy: List[str]):
    """Define subplots to display according to user configuration."""
    mode = cfg.grouping.get("mode", "all")
    if mode == "month":
        labels = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        indices = [_month_indices(m, dates) for m in labels]
        return labels, indices, 3, 4, 'year'
    if mode == "days":
        ndays = int(cfg.grouping.get("ndays", 3))
        idx_list = list(range(min(ndays, (len(dates) if dates is not None else ndays))))
        labels = [dates[i].strftime('%Y-%m-%d') if dates is not None else f'Day {i+1}' for i in idx_list]
        indices = [np.array([i]) for i in idx_list]
        return labels, indices, 1, len(labels), 'days'
    if mode == "custom":
        labels = cfg.grouping.get("labels", None)
        indices = cfg.grouping.get("indices", None)
        if labels is None or indices is None:
            raise ValueError("For grouping.mode='custom', provide both 'labels' and 'indices' in cfg.grouping.")
        ncols = len(labels)
        return labels, indices, 1, ncols, 'custom'
    # fallback to legacy months_to_plot if provided
    labels = months_to_plot_legacy if months_to_plot_legacy else ["ALL"]
    indices = [_month_indices(m, dates) for m in labels]
    return labels, indices, 1, len(labels), 'custom'


def _select_top_edges(edge_index, edge_mask, vis_mode, base_ew, ew_map, ew_min, ew_den, edge_keep_ratio):
    """Filter and weight edges according to vis_mode and ratio."""
    edges_all, weights_all = [], []
    for (u, v), m_val in zip(edge_index.T, edge_mask):
        m_val = float(m_val)
        if not np.isfinite(m_val) or m_val <= 0:
            continue
        if vis_mode == "context" and base_ew is not None:
            ew_val = ew_map.get((u, v), ew_map.get((v, u), 0.0))
            ew_norm = (ew_val - ew_min) / (ew_den + 1e-12)
            w = m_val * ew_norm
        else:
            w = m_val
        if np.isfinite(w):
            edges_all.append((int(u), int(v)))
            weights_all.append(float(w))
    if not weights_all:
        return [], []
    K = max(1, int(np.ceil(edge_keep_ratio * len(weights_all))))
    order = np.argsort(-np.asarray(weights_all))[:K]
    edges_draw = [edges_all[i] for i in order]
    weights_draw = [weights_all[i] for i in order]
    return edges_draw, weights_draw


def _draw_graph_panels(model_name, dataset_name, cfg, vis_mode, panel_data, G_base,
                       map_kwargs, vmin, vmax, nrows, ncols, tag, output_root="interpretability"):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    cmap = sns.cm.rocket_r if vis_mode != "std" else sns.cm.mako
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    for ax, (lbl, edges_draw, weights_draw) in zip(axes, panel_data):
        G = G_base.copy()
        for (u_i, v_i), w in zip(edges_draw, weights_draw):
            G.add_edge(u_i, v_i, weight=w)

        pos = nx.get_node_attributes(G, 'pos')
        bm_kwargs = dict(map_kwargs) if map_kwargs is not None else {}
        m = Basemap(resolution=cfg.map_resolution, ax=ax, **bm_kwargs)
        if cfg.draw_coastlines: m.drawcoastlines()
        if cfg.draw_countries: m.drawcountries()
        m.drawmapboundary(fill_color=cfg.mapboundary_color)
        m.fillcontinents(color=cfg.fillcontinents_color)

        pos_bm = {node: m(lon, lat) for node, (lon, lat) in pos.items()}
        node_list = list(pos_bm.keys())

        if cfg.show_nodes and node_list:
            nx.draw_networkx_nodes(G, pos_bm, nodelist=node_list,
                                   node_size=cfg.node_size, node_color=cfg.node_color,
                                   alpha=cfg.node_alpha, ax=ax)
        if cfg.show_labels and node_list:
            labels = {n: str(n) for n in node_list}
            nx.draw_networkx_labels(G, pos_bm, labels=labels,
                                    font_size=cfg.label_fontsize, font_color=cfg.label_color, ax=ax)

        if edges_draw:
            edgelist, edge_colors, widths = [], [], []
            for (u, v), w in zip(edges_draw, weights_draw):
                if u in pos_bm and v in pos_bm:
                    edgelist.append((u, v))
                    edge_colors.append(sm.to_rgba(w))
                    if np.isfinite(vmax) and (vmax > vmin):
                        t = (w - vmin) / (vmax - vmin + 1e-12)
                        t = max(0.0, min(1.0, t))
                    else:
                        t = 0.5
                    widths.append(cfg.edge_width_min + t * (cfg.edge_width_max - cfg.edge_width_min))
            if edgelist:
                use_arrows = cfg.edge_arrows and (len(edgelist) <= cfg.arrow_max_edges)
                if use_arrows:
                    nx.draw_networkx_edges(
                        G, pos_bm, ax=ax,
                        edgelist=edgelist,
                        edge_color=edge_colors,
                        alpha=cfg.edge_alpha,
                        width=widths,
                        arrows=True,
                        arrowstyle=cfg.arrowstyle,
                        arrowsize=cfg.arrowsize,
                        connectionstyle=cfg.connectionstyle,
                    )
                else:
                    nx.draw_networkx_edges(
                        G, pos_bm, ax=ax,
                        edgelist=edgelist,
                        edge_color=edge_colors,
                        alpha=cfg.edge_alpha,
                        width=widths,
                    )
            else:
                ax.text(0.5, 0.5, 'No edges', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(lbl, fontsize=max(10, ax.title.get_fontsize()))

    for ax in axes[len(panel_data):]:
        ax.axis('off')

    fig.suptitle(f'Explanation Graph ({vis_mode}) — {model_name}', fontsize=cfg.fontsize)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Colorbar closer (smaller pad) and with larger fonts
    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        shrink=0.8,
        aspect=30,
        pad=0.02,        # bring colorbar closer to subfigures
        fraction=0.04    # reasonable width for the colorbar
    )
    cbar.set_label({
        "std": "Edge importance variability (σ)",
        "context": "Edge importance (mask × normalized edge_weight)",
    }.get(vis_mode, "Edge importance (mean edge_mask)"), fontsize=cfg.fontsize)
    cbar.ax.tick_params(labelsize=cfg.labelsize)  

    out_dir = os.path.join(output_root, dataset_name, 'explanation_graph', vis_mode)
    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, f'{model_name}__{tag}.{cfg.file_ext}')
    plt.savefig(map_path, dpi=150)
    plt.close()
    print(f"Saved explanation grid for {model_name} ({vis_mode}, {dataset_name}) → {map_path}")
