import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Dict, Any, Tuple, Union, Optional
from transformers import AutoModel
from model.utils import freeze_bert_layers

class BaseModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any], dataset_config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.device = device
        
        self.plm = self._init_plm()
        
        freeze_layers = model_config.get('freeze_bert_layers', 0)
        freeze_bert_layers(self.plm, freeze_layers)

    def _init_plm(self):
        return AutoModel.from_pretrained(self.dataset_config['pretrained_model_name'])

    @abstractmethod
    def forward(self, candidate_news, clicked_news, clicked_news_mask) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def get_news_vector(self, news) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get news representation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_vector(self, clicked_news_vector, mask=None) -> torch.Tensor:
        """
        Get user representation.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_prediction(self, news_vector, user_vector) -> torch.Tensor:
        """
        Get click prediction from vectors.
        """
        raise NotImplementedError
