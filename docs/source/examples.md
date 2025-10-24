# Get started

## Adapt GraphToolbox to your problem

GraphToolbox is adapted to electricity forecasting, but you can use it for any regression task on a graph. Here are the steps to reproduce to use GraphToolbox for your usecase:

1. Put the datasets `train.csv` and `test.csv` in `lib/graphtoolbox/data/`.

2. Modify the file `config.py` in `lib/graphtoolbox/` which parameters your datasets and your models:
    - `model_kwargs`: parameters of your model,
    - `data_kwargs`: parameters of the `Data` object. `node_var` is the name of the node variable (in the example, it is `Region`). `dummies` are the dummy variables that you want to encode (the default dummies are categorical variables),
    - `dataset_kwargs`: parameters of the `GraphDataset` object. Here you can define the variable you want to predict and the covariates you want to use (including the dummy variables). In the example, the target variable is `load`.
    - `optim_kwargs`: parameters of the `Optimizer` object. Here you can define the grid of optimization.
    - `explain_kwargs`: parameters of the explainer.
    - `df_pos`: dataframe with geographical positions of your nodes. It is important to change this dataframe if you want to construct a spatial matrix, and if you want to use the explainer.

## Train a model 

Here is a basic example of how to train a GNN model with the `Trainer` class.

```python
from graphtoolbox.data.dataset import *
from graphtoolbox.training.trainer import Trainer
from graphtoolbox.utils.helper_functions import *
from torch_geometric.nn.models import *

# Load config
change_cwd(target_directory='GraphToolbox')
data_kwargs, dataset_kwargs, model_kwargs = load_configs()

# Load datasets
data = DataClass()
graph_dataset_train = GraphDataset(data=data, period='train')
graph_dataset_val = GraphDataset(data=data, period='val', scalers_feat=graph_dataset_train.scalers_feat, scalers_target=graph_dataset_train.scalers_target)
graph_dataset_test = GraphDataset(data=data, period='test', scalers_feat=graph_dataset_train.scalers_feat, scalers_target=graph_dataset_train.scalers_target, batch_size=len(data.df_test)//12)

# Initialize model
model = GATc(in_channels=graph_dataset_train.num_node_features, **model_kwargs) 

# Initialize trainer
trainer = Trainer(model=model, 
                  dataset_train=graph_dataset_train,
                  dataset_val=graph_dataset_val,
                  dataset_test=graph_dataset_test)

# Train model
pred_model_test, target_test, edge_index, attention_weights = trainer.train(num_epochs=100)

# Evaluate model
trainer.evaluate()
```

You can also train a model directly through the terminal with the following command (make sure to be at the root of the directory):

```sh
python lib/graphtoolbox/main.py --mode train
```

## Optimize a model

Here is a basic example of how to optimize a GNN model with the `Optimizer` class.

```python
from graphtoolbox.data.dataset import *
from graphtoolbox.optim.optimizer import Optimizer
from graphtoolbox.utils.helper_functions import *
from torch_geometric.nn.models import *

# Load config
change_cwd(target_directory='GraphToolbox')
data_kwargs, dataset_kwargs, model_kwargs = load_configs()

# Load datasets
data = DataClass()
graph_dataset_train = GraphDataset(data=data, period='train')
graph_dataset_val = GraphDataset(data=data, period='val', scalers_feat=graph_dataset_train.scalers_feat, scalers_target=graph_dataset_train.scalers_target)
graph_dataset_test = GraphDataset(data=data, period='test', scalers_feat=graph_dataset_train.scalers_feat, scalers_target=graph_dataset_train.scalers_target, batch_size=len(data.df_test)//12)

# Initialize optimizer
optimizer = Optimizer(model=GraphSAGE, 
                      dataset_train=graph_dataset_train,
                      dataset_val=graph_dataset_val,
                      num_epochs=200)

# Optimize model
optimizer.optimize()

# (Optional) Display the results on a dashboard
optimizer.run_on_server()
```

You can also optimize a model directly through the terminal with the following command (make sure to be at the root of the directory):

```sh
python lib/graphtoolbox/main.py --mode optim
```


## Evaluate a list of models

The evaluation of multiple models can be done directly through the terminal (make sure to be at the root of the directory). You can use the following command to get a `.csv` file with the results of your models:

```sh
python lib/graphtoolbox/training/evaluation.py --optim 1
```

Here `--optim 1` indicates that the evaluated models are in `checkpoints_optim/`. If you prefer to evaluate the mdoels in `checkpoints/`, then change the flag to 0.

## Explain a model

Documentation incoming ! 