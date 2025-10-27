import graphtoolbox.training.metrics
from graphtoolbox.utils.helper_functions import *
from graphtoolbox.utils.visualizations import *
import numpy as np
import os
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from typing import List, Tuple, Union

# Set device to 'cuda' if available, 'mps' if on MACOS, else 'cpu'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float, default=0.0
        Minimum change in validation loss to qualify as an improvement.

    Attributes
    ----------
    counter : int
        Number of consecutive epochs without improvement.
    best_loss : float
        Lowest recorded validation loss.
    early_stop : bool
        Whether the early stopping condition has been met.

    Examples
    --------
    >>> stopper = EarlyStopping(patience=5, min_delta=0.01)
    >>> for epoch in range(100):
    ...     val_loss = compute_validation_loss()
    ...     stopper(val_loss)
    ...     if stopper.early_stop:
    ...         print("Stopped early at epoch", epoch)
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Update early stopping state with the latest validation loss.

        Parameters
        ----------
        val_loss : float
            Current validation loss for this epoch.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    """
    Train, validate, and evaluate a graph neural network model on temporal graph datasets.

    This class handles the full training loop, including:
    - batched training with PyTorch Geometric loaders,
    - validation and early stopping,
    - checkpointing and loss tracking,
    - inference and hierarchical reconciliation (MinT).

    Parameters
    ----------
    model : torch.nn.Module
        Graph neural network model to train.
    dataset_train : GraphDataset
        Training dataset.
    dataset_val : GraphDataset
        Validation dataset.
    dataset_test : GraphDataset
        Test dataset.
    batch_size : int
        Number of graph samples per batch.
    model_kwargs : dict, optional
        Dictionary of model hyperparameters (loaded from config if None).
    reconcile : bool, default=True
        Whether to apply MinT reconciliation to predictions.
    **kwargs :
        Optional arguments:
            - edge_index (torch.Tensor)
            - edge_weight (torch.Tensor)
            - return_attention (bool)
            - lam_reg (float): regularization weight.

    Attributes
    ----------
    is_trained : bool
        Whether the model has been trained.
    train_loader, val_loader, test_loader : PyGDataLoader
        Dataloaders for training, validation, and test.
    saving_directory : str
        Path to saved model checkpoints.
    S, G, P : torch.Tensor
        Matrices for hierarchical MinT reconciliation.
    """
    def __init__(self, model, dataset_train, dataset_val, dataset_test, batch_size,
                 model_kwargs: Optional[Dict] = None, reconcile: bool = True,
                 **kwargs):
        self.edge_index = kwargs.get('edge_index', None)
        self.edge_weight = kwargs.get('edge_weight', None)
        self.return_attention = kwargs.get('return_attention', False)
        self.lam_reg = kwargs.get('lam_reg', 0)
        self.model = model.to(DEVICE)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.reconcile = reconcile
        self.nodes = self.dataset_train.nodes
        self.num_nodes = self.dataset_train.num_nodes
        self.folder_config = dataset_train.data.folder_config

        dataset_kwargs = dataset_train.dataset_kwargs
        if dataset_kwargs is None:
            dataset_kwargs = load_kwargs(folder_config=self.folder_config, kwargs='dataset_kwargs')
        self.dataset_kwargs = dataset_kwargs

        if model_kwargs is None:
            model_kwargs = load_kwargs(folder_config=self.folder_config, kwargs='model_kwargs')
        self.model_kwargs = model_kwargs
        self.is_trained = False

        self.batch_size = batch_size
        self.train_loader = PyGDataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = PyGDataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = PyGDataLoader(dataset_test, shuffle=False, drop_last=False)

        self._build_summing_matrix()
        self._compute_min_trace_projection()

    def train(self, **kwargs) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Train the model and optionally evaluate during training.

        Supports early stopping, checkpoint saving, and attention visualization.

        Parameters
        ----------
        num_epochs : int, optional
            Number of epochs to train (default: from model_kwargs).
        optimizer : torch.optim.Optimizer, optional
            Optimizer (default: Adam with model_kwargs['lr']).
        patience : int, default=20
            Early stopping patience.
        min_delta : float, default=0.0
            Minimum delta to count as improvement.
        force_training : bool, default=False
            If True, retrains even if checkpoint exists.
        saving_directory : str, optional
            Folder to store model weights.
        plot_loss : bool, optional
            If True, plots training/validation curves.
        dynamic_graph : bool, optional
            Enable dynamic adjacency matrix updates per epoch.
        save : bool, optional
            If True, saves attention maps.

        Returns
        -------
        preds : np.ndarray
            Rescaled model predictions on the test set.
        targets : torch.Tensor
            Ground-truth values.
        edge_index : torch.Tensor
            Graph connectivity.
        Lattention_mat or edge_weight : torch.Tensor
            Attention matrices (if applicable).
        """
        self.num_epochs = kwargs.get('num_epochs', self.model_kwargs['num_epochs'])
        optimizer = kwargs.get('optimizer', torch.optim.Adam(self.model.parameters(), lr=self.model_kwargs['lr']))
        force_training = kwargs.get('force_training', False)
        patience = kwargs.get('patience', 20)
        min_delta = kwargs.get('min_delta', 0.0)
        self.dynamic_graph = kwargs.get('dynamic_graph', False)
        save = kwargs.get('save', False)
        
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        if hasattr(self.model, 'conv_class'):
            self.model_name = self.model.conv_class.__name__
        else:
            self.model_name = self.model.__class__.__name__
        self.hidden_channels = self.model.hidden_channels
        self.num_layers = self.model.num_layers
        self.adj_matrix = self.dataset_train.adj_matrix
        Lattention_mat = []
        try:
            self.heads = self.model.heads
        except:
            self.heads = 0
        saving_directory = kwargs.get('saving_directory',
                                    f'./checkpoints/{self.model_name}_{self.adj_matrix}/batch{self.batch_size}_hidden{self.hidden_channels}_layers{self.num_layers}_epochs{self.num_epochs}')
        if hasattr(self.model, 'conv_kwargs'):
                for k, v in self.model.conv_kwargs.items():
                    saving_directory += f'_{k}{v}'
        self.saving_directory = saving_directory
        if not os.path.exists(saving_directory) or len(os.listdir(saving_directory)) == 0 or force_training:
            print("Training model...")
            os.makedirs(saving_directory, exist_ok=True)
            clean_dir(saving_directory)
            train_losses = []
            val_losses = []
            best_loss = float('inf')
            num_epochs_final = self.num_epochs
            for epoch in tqdm(range(self.num_epochs)):
                params_filename = 'epoch{}.params'.format(epoch)
                train_loss = self._run_epoch(optimizer, 'train', self.train_loader, return_attention=False)
                if self.return_attention:
                    val_loss, attention_mat = self._run_epoch(optimizer, 'eval', self.val_loader, return_attention=self.return_attention)
                    Lattention_mat.append(attention_mat)
                else:
                    val_loss = self._run_epoch(optimizer, 'eval', self.val_loader)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    clean_dir(saving_directory)
                    torch.save(self.model.state_dict(), os.path.join(saving_directory, params_filename))

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    num_epochs_final = epoch + 1
                    break

            if kwargs.get('plot_loss', False):
                plot_losses(num_epochs_final, train_losses, val_losses)
        else:
            print("Loading pretrained model.")
            self.model.load_state_dict(torch.load(os.path.join(saving_directory, os.listdir(saving_directory)[0]), map_location=DEVICE))

        self.batch_size_save = kwargs.get('batch_size_save', self.batch_size)
        self.test_loader = PyGDataLoader(self.dataset_test, shuffle=False, drop_last=False)
        if self.return_attention and save:
            self.val_loader = PyGDataLoader(self.dataset_val, batch_size=self.batch_size_save, shuffle=False)
            self.train_loader = PyGDataLoader(self.dataset_train, batch_size = self.batch_size_save, shuffle=False)
            _ = self._run_epoch(optimizer, 'eval', self.train_loader, return_attention=self.return_attention, save=True, dataset_name='train')
            _ = self._run_epoch(optimizer, 'eval', self.val_loader, return_attention=self.return_attention, save=True, dataset_name='val')
            _ = self._run_epoch(optimizer, 'eval', self.test_loader, return_attention=self.return_attention, save=True, dataset_name='test')

        _, preds, targets = self._predict(self.test_loader)
        self.is_trained = True
        if self.return_attention :
            self.return_attention = False
            return (preds, targets, self.edge_index, Lattention_mat) 
        else :
            return (preds, targets, self.edge_index, self.edge_weight)

    def _run_epoch(self, optimizer, mode: str, loader: PyGDataLoader, return_attention: bool = False, save: bool = False, dataset_name: str = 'test') -> float:
        """
        Run a single training or evaluation epoch.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer used for parameter updates (only when mode='train').
        mode : {'train', 'eval'}
            Whether to update weights or evaluate.
        loader : PyGDataLoader
            DataLoader providing batched graph data.
        return_attention : bool, default=False
            Whether to collect and return attention weights.
        save : bool, default=False
            Whether to save attention maps on disk.
        dataset_name : str, default='test'
            Dataset label for saved outputs.

        Returns
        -------
        float
            Average loss for the epoch.
        """
        assert mode in ['train', 'eval']
        num_nodes = self.dataset_train.num_nodes
        self.model.train() if mode == 'train' else self.model.eval()
        total_loss, count = 0.0, 0

        if save: 
            save_path = f"./attention_matrix/{self.model_name}_{self.adj_matrix}/{dataset_name}_batch{self.batch_size}_hidden{self.hidden_channels}_layers{self.num_layers}_epochs{self.num_epochs}_heads{self.heads}"
            self.save_path_MA = save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            clean_dir(save_path)

        # TODO: add dynamic graph condition

        for i, batch in enumerate(loader):
            # Ensure float32 dtypes before moving to DEVICE
            for attr in ['x', 'y_scaled', 'y', 'edge_weight', 'mask_y']:
                if getattr(batch, attr, None) is not None:
                    setattr(batch, attr, getattr(batch, attr).to(torch.float32))
            batch = batch.to(DEVICE)
            if torch.isnan(batch.x).any() or torch.isnan(batch.y_scaled).any():
                print(f"[WARN] NaN detected in batch {i}. Skipping batch.")
                continue

            if (mode == 'eval')  and (return_attention): 
                if save: 
                    save_path = f"./attention_matrix/{self.model_name}_{self.adj_matrix}/{dataset_name}_batch{self.batch_size}_hidden{self.hidden_channels}_layers{self.num_layers}_epochs{self.num_epochs}_heads{self.heads}/num_batch{i}.pt"
                    out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None), mask=getattr(batch, 'mask_y', None), return_attention=True, batch_size=self.batch_size_save, save=True , save_path=save_path)
                    out = out.squeeze().view(-1, num_nodes).T
                else:
                    if i==0:
                        out, tot_dict_attention = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None), mask=getattr(batch, 'mask_y', None), return_attention=True, batch_size=self.batch_size)
                        out = out.squeeze().view(-1, num_nodes).T
                    elif (i > 0) and (i < (len(loader)-1)):
                        out, dict_attention = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None), mask=getattr(batch, 'mask_y', None), return_attention=True, batch_size=self.batch_size)
                        out = out.squeeze().view(-1, num_nodes).T
                        for k in range(len(tot_dict_attention['mean'])) :
                            tot_dict_attention['mean'][k] += dict_attention['mean'][k]
                            tot_dict_attention['std'][k] += dict_attention['std'][k]
                        del dict_attention
                    else:
                        out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None), mask=getattr(batch, 'mask_y', None)).squeeze().view(-1, num_nodes).T
            else:
                # TODO: add dynamic graph condition
                out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None), mask=getattr(batch, 'mask_y', None)).squeeze().view(-1, num_nodes).T           
 
            y_s = batch.y_scaled.view(-1, num_nodes).T
            mask = batch.mask_y.view(-1, num_nodes).T  
            if mask.sum() > 0:
                mse_loss = torch.sum(((out - y_s) ** 2) * mask) / mask.sum()
            else:
                print(f"[WARN] Batch {i} ignoré car aucune cible valide")
                continue
            # mse_loss = torch.mean((out - y_s) ** 2)
            pred_diff = out[:, None, :] - out[None, :, :]
            norms = torch.norm(pred_diff, p=2, dim=2)
            reg_loss = norms.mean()
            loss = mse_loss + self.lam_reg * reg_loss
            del y_s
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
 
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

        if (mode == 'eval')  and (return_attention) and (len(loader) > 2) and (not save) :
            for k in range(len(tot_dict_attention['mean'])) :
                tot_dict_attention['mean'][k] /= len(loader)
                tot_dict_attention['std'][k] /= len(loader)

        if (return_attention) and (not save):
            if count > 0:
                return (total_loss / count, tot_dict_attention) 
            else:
                return (0, tot_dict_attention)
        else:
            if count > 0:
                return total_loss / count
            else:
                return 0

    def _predict(self, loader: PyGDataLoader) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Run inference on a dataset and compute prediction loss.

        Parameters
        ----------
        loader : PyGDataLoader
            DataLoader for evaluation.

        Returns
        -------
        loss : float
            RMSE between predictions and targets.
        preds : torch.Tensor
            Rescaled predictions (num_nodes × T).
        targets : torch.Tensor
            Ground-truth targets (num_nodes × T).
        """
        self.model.eval()
        num_nodes = self.dataset_train.num_nodes
        y_preds, y_targets = [], []
        with torch.no_grad():
            for batch in loader:
                for attr in ['x', 'y_scaled', 'y', 'edge_weight', 'mask_y']:
                    if getattr(batch, attr, None) is not None:
                        setattr(batch, attr, getattr(batch, attr).to(torch.float32))
                batch = batch.to(DEVICE)
                if hasattr(batch, 'edge_weight') and batch.edge_weight is not None:
                    batch.edge_weight = batch.edge_weight.float()
                out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None), mask=getattr(batch, 'mask_y', None))
                y_preds.append(out)
                y_targets.append(batch.y.cpu().detach())
        y_targets = torch.hstack(y_targets) 
        y_targets = y_targets.reshape(num_nodes, -1)
        y_preds = torch.hstack(y_preds)  
        y_preds = y_preds.reshape(num_nodes, -1)[:, :y_targets.shape[1]]
        pred_rescaled = self._rescale_predictions(y_preds).cpu().detach()
        del y_preds
        if self.reconcile:
            pred_rescaled = self._min_trace_reconciliation(preds=pred_rescaled).cpu().detach()
            pred_rescaled = pred_rescaled[:-1] * self.dataset_test.mask_Y[:y_targets.shape[1],:y_targets.shape[1]]
        else:
            pred_rescaled = pred_rescaled * self.dataset_test.mask_Y
        loss = getattr(graphtoolbox.training.metrics, 'RMSE')(preds=pred_rescaled.cpu().detach().sum(dim=0), targets=y_targets.sum(dim=0)).item()
        return loss, pred_rescaled, y_targets

    def evaluate(self, losses: Union[List[str], str] = ['mape', 'rmse']):
        """
        Evaluate trained model on test set using given metrics.

        Parameters
        ----------
        losses : str or list of str, default=['mape', 'rmse']
            Metrics to compute. Supported: 'mape', 'rmse'.

        Returns
        -------
        None
            Prints evaluation metrics.
        """
        if not self.is_trained:
            print("You need to train the model first!")
            return

        _, preds, targets = self._predict(self.test_loader)

        if isinstance(losses, str):
            losses = [losses]

        for loss in losses:
            try:
                eps = 1e-6
                loss_fn = getattr(graphtoolbox.training.metrics, loss.upper())
                result = loss_fn(preds=preds.cpu().detach().sum(dim=0) + eps, targets=targets.sum(dim=0).cpu().detach() + eps)
                unit = "%" if loss.lower() == 'mape' else "MW"
                val = result.item() * 100 if loss.lower() == 'mape' else result.item()
                print(f"{loss.upper()} on test set: {val:.4f} {unit}")
            except AttributeError:
                print(f"Loss function {loss} not found.")
        
    def _rescale_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform model predictions using stored target scalers.

        Parameters
        ----------
        preds : torch.Tensor
            Normalized predictions (num_nodes × T).

        Returns
        -------
        torch.Tensor
            Rescaled predictions (num_nodes × T) in original units (float32).
        """
        preds_rescaled = []
        for node_idx, node in enumerate(self.nodes):
            scaler = self.dataset_train.scalers_target[node]
            pred_np = preds[node_idx].detach().cpu().numpy().reshape(-1, 1)
            pred_rescaled_np = scaler.inverse_transform(pred_np).reshape(-1)
            preds_rescaled.append(torch.as_tensor(pred_rescaled_np, dtype=torch.float32))
        return torch.stack(preds_rescaled, dim=0).to(dtype=torch.float32, device=preds.device)
            
    def _build_summing_matrix(self) -> torch.Tensor:
        """
        Build the summing matrix S for hierarchical aggregation.

        Returns
        -------
        torch.Tensor
            Structure matrix combining base and total nodes.
        """
        I = torch.eye(self.num_nodes)  
        total = torch.ones((1, self.num_nodes))
        self.S = torch.cat([I, total], dim=0).to(DEVICE)
    
    def _compute_min_trace_projection(self, W: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the MinT projection matrix for hierarchical reconciliation.

        Parameters
        ----------
        W : torch.Tensor, optional
            Weight matrix (defaults to identity).

        Returns
        -------
        torch.Tensor
            Projection matrix P such that reconciled forecasts = S @ G @ forecasts.
        """
        if W is None:
            W_inv = torch.eye(self.S.shape[0], device=DEVICE)
            self.W = W_inv
        else:
            self.W = W.to(DEVICE)
            if W.shape[0] == W.shape[1] and torch.allclose(W, torch.diag(torch.diagonal(W))):
                W_inv = torch.diag(1.0 / torch.diagonal(W))
            else:
                W_inv = torch.inverse(W)

        S_t = self.S.T
        middle = torch.inverse(S_t @ W_inv @ self.S)
        self.G = middle @ S_t @ W_inv
        self.P = self.S @ self.G
 
    def _min_trace_reconciliation(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Apply MinT reconciliation to hierarchical forecasts.

        Parameters
        ----------
        preds : torch.Tensor
            Model forecasts for base series.

        Returns
        -------
        torch.Tensor
            Reconciled forecasts (base + aggregated).
        """
        national_pred = preds.sum(axis=0).unsqueeze(0)
        new_preds = torch.cat([preds, national_pred]).to(DEVICE)
        return self.S @ self.G @ new_preds