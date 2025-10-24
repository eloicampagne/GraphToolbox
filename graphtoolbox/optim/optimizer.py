from graphtoolbox.data.dataset import *
from graphtoolbox.utils.helper_functions import *
import datetime
import logging
import optuna
from optuna_dashboard import run_server
import sys
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn.conv import GATv2Conv
from tqdm import tqdm

class Optimizer():
    """
    Hyperparameter optimizer for graph neural networks using Optuna.

    The `Optimizer` class automates hyperparameter tuning of GNN models
    on training and validation datasets. It supports structured logging,
    pruning of poor trials, and optional dashboard visualization via
    `optuna-dashboard`.

    Parameters
    ----------
    model : torch.nn.Module
        GNN model class to be optimized (not an instance).
    dataset_train : GraphDataset
        Training dataset.
    dataset_val : GraphDataset
        Validation dataset.
    optim_kwargs : dict, optional
        Search space definition for hyperparameters.
        Example: ``{"hidden_channels": (32, 128), "num_layers": (2, 5), "lr": (1e-4, 1e-2)}``.
        Loaded from the configuration folder if not provided.
    num_epochs : int, optional
        Number of epochs to train each trial. Default is 200.
    conv_class : torch_geometric.nn.MessagePassing, optional
        Convolution class to use in the model (default: ``GATv2Conv``).

    Attributes
    ----------
    study : optuna.Study
        Optuna study object containing all trials and results.
    storage : optuna.storages.InMemoryStorage
        In-memory storage backend for optimization results.
    is_optimized : bool
        Whether the optimization process has been executed.
    logger : logging.Logger
        Logger instance for progress and diagnostic output.

    Examples
    --------
    >>> opt = Optimizer(model=myGNN, dataset_train=train_set, dataset_val=val_set)
    >>> opt.optimize(n_trials=30)
    >>> opt.run_on_server()  # visualize results
    """
    def __init__(self, model, dataset_train: GraphDataset, dataset_val: GraphDataset, optim_kwargs: Dict = None, **kwargs):
        self.model_class = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        if optim_kwargs is None:
            self.optim_kwargs = load_kwargs(folder_config=dataset_train.folder_config, kwargs='optim_kwargs')
        else:
            self.optim_kwargs = optim_kwargs
        self.num_epochs = kwargs.get('num_epochs', 200)
        self.conv_class = kwargs.get('conv_class', GATv2Conv)
        self.is_optimized = False
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f'optimization_{str(datetime.date.today())}.log')
        logging.basicConfig(filename=filename, level='INFO')
        self.logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Init logger.')
        self.logger.info(f'Optimizer is initialized for model {self.model_class.__name__}.')

    def _run_epoch(self, optimizer, mode: str, loader: PyGDataLoader) -> float:
        """
        Run a single epoch of training or evaluation.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer instance (used only when `mode='train'`).
        mode : {'train', 'eval'}
            Operating mode for model update or evaluation.
        loader : torch_geometric.loader.DataLoader
            Data loader providing graph batches.

        Returns
        -------
        float
            Average RMSE loss for the epoch.

        Notes
        -----
        - Skips batches containing NaN values in features or targets.
        - Applies gradient clipping to stabilize training.
        """
 
        assert mode in ['train', 'eval']
        num_nodes = self.dataset_train.num_nodes
        self.model.train() if mode == 'train' else self.model.eval()
        total_loss, count = 0.0, 0
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            batch.x = batch.x.float()
            if hasattr(batch, 'edge_weight') and batch.edge_weight is not None:
                batch.edge_weight = batch.edge_weight.float()
            if torch.isnan(batch.x).any() or torch.isnan(batch.y_scaled).any():
                print(f"[WARN] NaN detected in batch {i}. Skipping batch.")
                continue
            
            out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None)).squeeze().view(-1, num_nodes).T           
 
            y_s = batch.y_scaled.view(-1, num_nodes).T
            mask = batch.mask_y.view(-1, num_nodes).T  
            if mask.sum() > 0:
                mse_loss = torch.sum(((out - y_s) ** 2) * mask) / mask.sum()
            else:
                print(f"[WARN] Batch {i} ignoré car aucune cible valide")
                continue
            loss = mse_loss
            del y_s, out
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
 
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

            if count > 0:
                return total_loss / count, optimizer
            else:
                return 0, optimizer

    def _define_model(self, trial):
        """
        Build a model instance for a given Optuna trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Current trial from which hyperparameters are sampled.

        Returns
        -------
        torch.nn.Module
            Initialized model with trial-specific parameters.
        """
        vars = {}
        for (param, (vmin, vmax)) in zip(self.optim_kwargs.keys(), self.optim_kwargs.values()):
            if param != 'batch_size':
                if isinstance(vmin, int):
                    vars[param] = trial.suggest_int(param, vmin, vmax)
                elif isinstance(vmin, float):
                    if param != 'lr':
                        vars[param] = trial.suggest_float(param, vmin, vmax, log=False)
                    else:
                        vars[param] = trial.suggest_float(param, vmin, vmax, log=True)
                        self.lr = vars[param]
                else:
                    # trial.suggest_categorical(param, categories)
                    print(param, vmin, vmax, type(vmin))
                    raise NotImplementedError()
            else:
                pass   
        model = self.model_class(in_channels=self.dataset_val.num_node_features, conv_class=self.conv_class, conv_kwargs=vars, out_channels=48, **vars)
        return model
    
    def _objective(self, trial):
        """
        Objective function evaluated by Optuna for each trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial used to sample hyperparameters.

        Returns
        -------
        float
            Validation RMSE (lower is better).

        Notes
        -----
        - Defines loaders for train/val sets.
        - Saves best-performing model state per trial.
        - Supports pruning based on intermediate results.
        """
        self.model = self._define_model(trial).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        adj_matrix = trial.suggest_categorical('adj_matrix', os.listdir(self.dataset_train.graph_folder))
        self.dataset_train._set_adj_matrix(adj_matrix=adj_matrix)
        self.dataset_val._set_adj_matrix(adj_matrix=adj_matrix)
        saving_directory = f'./checkpoints_optim/{self.model.__class__.__name__}{self.model.heads}_{self.dataset_train.adj_matrix}/batch{batch_size}_hidden{self.model.hidden_channels}_layers{self.model.num_layers}_epochs{self.num_epochs}'

        os.makedirs(saving_directory, exist_ok=True)
        train_loader = PyGDataLoader(self.dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(self.num_epochs)):
            params_filename = 'epoch{}.params'.format(epoch)
            train_loss, optimizer = self._run_epoch(optimizer=optimizer, mode='train', loader=train_loader)
            train_losses.append(train_loss)
            val_loss, _ = self._run_epoch(optimizer=optimizer, mode='eval', loader=val_loader)
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                clean_dir(saving_directory)
                torch.save(self.model.state_dict(), os.path.join(saving_directory, params_filename))
            trial.report(best_loss, epoch)
            if trial.should_prune():
                shutil.rmtree(saving_directory)
                raise optuna.exceptions.TrialPruned()
        return best_loss
    
    def optimize(self, **kwargs):
        """
        Run the Optuna optimization loop.

        Parameters
        ----------
        study_name : str, optional
            Name for the Optuna study. Default is derived from the model class.
        n_trials : int, default=100
            Number of hyperparameter trials to perform.
        direction : {'minimize', 'maximize'}, default='minimize'
            Optimization direction for the objective function.
        timeout : int, optional
            Maximum runtime in seconds.

        Side Effects
        ------------
        - Saves best parameters and statistics to `./results_optim_<ConvClass>/`.
        - Logs progress to `./logs/optimization_<date>.log`.

        Notes
        -----
        Uses in-memory storage by default but can be extended for database-backed
        studies if persistence is needed.
        """
        self.storage = optuna.storages.InMemoryStorage()
        self.is_optimized = True
        self.study = optuna.create_study(storage=self.storage,
                                         study_name=kwargs.get('study_name', f'{self.model_class.__name__}_hpo'),
                                         direction=kwargs.get('direction', 'minimize'))
        self.logger.info('Optimization began.')
        self.study.optimize(self._objective, n_trials=kwargs.get('n_trials', 100), timeout=kwargs.get('timeout', 10000))
        self.logger.info('Optimization finished.')
        self.pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        self.complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(self.pruned_trials))
        print("  Number of complete trials: ", len(self.complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        result_dir = f'./results_optim_{self.conv_class.__name__}'
        result_file = os.path.join(result_dir, f'results_{self.model_class.__name__}.txt')
        os.makedirs(result_dir, exist_ok=True)

        with open(result_file, 'a') as f:
            f.write("Study statistics:\n")
            f.write(f"  Number of finished trials: {len(self.study.trials)}\n")
            f.write(f"  Number of pruned trials: {len(self.pruned_trials)}\n")
            f.write(f"  Number of complete trials: {len(self.complete_trials)}\n\n")

            f.write("Best trial:\n")
            f.write(f"  Value: {trial.value}\n\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")

    def run_on_server(self):
        """
        Launch an interactive Optuna dashboard to visualize study results.

        Requires that `optimize()` has already been executed.

        Raises
        ------
        RuntimeError
            If called before the optimization is completed.
        """
        if self.is_optimized:
            run_server(self.storage)
        else:
            print('You need to optimize your model first!')