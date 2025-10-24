![alt text](docs/source/graphtoolbox.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Maintainer](https://img.shields.io/badge/maintainer-E.Campagne-blue)

# GraphToolbox

GraphToolbox is a Python package designed for graph machine learning focused on electricity load forecasting. It provides tools for data handling, model building, training, evaluation, and visualization.

## Features

- Data handling and preprocessing for graph datasets.
- Various graph neural network models including Graph Convolutional Networks (GCNs), GraphSAGE and Graph Attention Networks (GATs).
- Training and evaluation utilities for graph-based models.
- Visualization tools for graph data and model results.

## Installation

To install GraphToolbox, clone the repository and install the dependencies:

```sh
git clone git@gitlab.pleiade.edf.fr:energy-data-lab/corporate/Theses/eloi-campagne/graphtoolbox.git
cd GraphToolbox
pip install .
```

## Usage

Here is a basic example of how to use GraphToolbox:

```python
from graphtoolbox.data.dataset import *
from graphtoolbox.training.trainer import Trainer
from graphtoolbox.utils.helper_functions import *
from torch_geometric.nn.models import *

# Load datasets
out_channels = 48
data = DataClass(path_train='./train.csv', 
                 path_test='./test.csv', 
                 data_kwargs=data_kwargs,
                 folder_config='.')

graph_dataset_train = GraphDataset(data=data, period='train', 
                                   graph_folder='../graph_representations',
                                   dataset_kwargs=dataset_kwargs,
                                   out_channels=out_channels)
graph_dataset_val = GraphDataset(data=data, period='val', 
                                 scalers_feat=graph_dataset_train.scalers_feat, 
                                 scalers_target=graph_dataset_train.scalers_target,
                                 graph_folder='../graph_representations',
                                 dataset_kwargs=dataset_kwargs,
                                 out_channels=out_channels)
graph_dataset_test = GraphDataset(data=data, period='test',
                                  scalers_feat=graph_dataset_train.scalers_feat, 
                                  scalers_target=graph_dataset_train.scalers_target,
                                  graph_folder='../graph_representations',
                                  dataset_kwargs=dataset_kwargs,
                                  out_channels=out_channels)

# Initialize model
conv_class = GATConv
conv_kwargs = {'heads': 2}
params = {'num_layers': 3, 
          'hidden_channels': 364, 
          'lr': 1e-3, 
          'batch_size': 16, 
          'adj_matrix': 'gl3sr', 
          'lam_reg': 0}

model = myGNN(
    in_channels=graph_dataset_train.num_node_features,
    num_layers=params["num_layers"],
    hidden_channels=params["hidden_channels"],
    out_channels=out_channels,
    conv_class=conv_class,
    conv_kwargs=conv_kwargs
)

# Initialize trainer
trainer = Trainer(
    model=model,
    dataset_train=graph_dataset_train,
    dataset_val=graph_dataset_val,
    dataset_test=graph_dataset_test,
    batch_size=params["batch_size"],
    return_attention=False,
    model_kwargs={'lr': params["lr"], 'num_epochs': 200},
    lam_reg=params["lam_reg"]
)

# Train model
pred_model_test, target_test, edge_index, attention_weights = trainer.train(
    plot_loss=True,
    force_training=True,
    save=False,
    patience=75
)

# Evaluate model
trainer.evaluate()
```

## Directory Structure

The main files and directories in GraphToolbox are as follows:
```
GraphToolbox/
├── examples/
│   ├── graph_representations/
│   │   ├── correlation/W.txt
│   │   ├── distsplines/W.txt
│   │   ├── distsplines2/W.txt
│   │   ├── dtw/W.txt
│   │   ├── eye/W.txt
│   │   ├── gl3sr/W.txt
│   │   ├── precision/W.txt
│   │   └── space/W.txt
│   ├── load/
│   │   ├── checkpoints/
│   │   ├── config.py
│   │   ├── example.ipynb
│   │   ├── train.csv
│   │   └── test.csv
│   ├── netload/
│   │   ├── checkpoints/
│   │   ├── config.py
│   │   ├── example.ipynb
│   │   ├── train.csv
│   │   └── test.csv
├── lib/
│   ├── graphtoolbox/
│   │   ├── data/
│   │   │   ├── dataset.py
│   │   │   └── preprocessing.py
│   │   ├── models/
│   │   │   └── gnn.py
│   │   ├── optim/
│   │   │   └── optimizer.py
│   │   ├── scripts/
│   │   │   ├── main.sh
│   │   │   ├── modify_config.py
│   │   │   ├── parallel.sh
│   │   │   └── run_mode.py
│   │   ├── training/
│   │   │   ├── evaluation.py
│   │   │   ├── metrics.py
│   │   │   └── trainer.py
│   │   ├── utils/
│   │   │   ├── attention.
│   │   │   ├── GL_3SR.py
│   │   │   ├── helper_functions.py
│   │   │   └── visualizations.py
│   │   ├── config.py
│   │   └── main.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1. Fork the project
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'feat: Add some feature'`. Please use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)!
4. Push to the branch: `git push origin feature/YourFeature`
5. Create a new Pull Request

Special **thanks** to all contributors of the GraphToolbox project:

- Eloi Campagne 
- Itai Zehavi

## License

This project is licensed under the MIT License - see the LICENSE file for details.