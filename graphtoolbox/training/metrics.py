import numpy as np
import torch
from typing import Union

def MAE(preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, float]:
    """
    Mean Absolute Error (MAE) between predictions and targets.

    Parameters:
        preds (Union[torch.Tensor, np.ndarray]): Predicted values.
        targets (Union[torch.Tensor, np.ndarray]): True values.

    Returns:
        Union[torch.Tensor, float]: The mean absolute error.
    """
    if isinstance(preds, torch.Tensor):
        return torch.mean(torch.abs(preds - targets))
    else:
        return np.mean(np.abs(preds - targets))
    
def NMAE(preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, float]:
    """
    Normalized Mean Absolute Error (NMAE) between predictions and targets.

    Parameters:
        preds (Union[torch.Tensor, np.ndarray]): Predicted values.
        targets (Union[torch.Tensor, np.ndarray]): True values.

    Returns:
        Union[torch.Tensor, float]: The normalized mean absolute error.
    """
    if isinstance(preds, torch.Tensor):
        return torch.mean(torch.abs(preds - targets)) / torch.mean(torch.abs(targets))
    else:
        return np.mean(np.abs(preds - targets)) / np.mean(np.abs(targets))

def MAPE(preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, float]:
    """
    Mean Absolute Percentage Error (MAPE) between predictions and targets.

    Parameters:
        preds (Union[torch.Tensor, np.ndarray]): Predicted values.
        targets (Union[torch.Tensor, np.ndarray]): True values.

    Returns:
        Union[torch.Tensor, float]: The mean absolute percentage error.
    """
    if isinstance(preds, torch.Tensor):
        return torch.mean(torch.abs((targets - preds) / targets))
    else:
        return np.mean(np.abs((targets - preds) / targets))

def RMSE(preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, float]:
    """
    Root Mean Square Error (RMSE) between predictions and targets.

    Parameters:
        preds (Union[torch.Tensor, np.ndarray]): Predicted values.
        targets (Union[torch.Tensor, np.ndarray]): True values.

    Returns:
        Union[torch.Tensor, float]: The root mean square error.
    """
    if isinstance(preds, torch.Tensor):
        return torch.sqrt(torch.mean(torch.square(targets - preds)))
    else:
        return np.sqrt(np.mean(np.square(targets - preds)))

def BIAS(preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, float]:
    """
    Bias between predictions and targets.

    Parameters:
        preds (Union[torch.Tensor, np.ndarray]): Predicted values.
        targets (Union[torch.Tensor, np.ndarray]): True values.

    Returns:
        Union[torch.Tensor, float]: The bias (mean error).
    """
    if isinstance(preds, torch.Tensor):
        return torch.mean(targets - preds)
    else:
        return np.mean(targets - preds)
