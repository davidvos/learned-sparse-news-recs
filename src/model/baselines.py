"""Neural dense baseline models.

Implementations of four widely-used neural news recommendation baselines:

- **NRMS**: Neural News Recommendation with Multi-Head Self-Attention.
- **NAML**: Neural News Recommendation with Attentive Multi-View Learning.
- **LSTUR**: Neural News Recommendation with Long- and Short-term User Representations.
- **NPA**: Neural News Recommendation with Personalized Attention.

All baselines in this file produce dense user and news vectors, relying on
FAISS exact MIPS for downstream retrieval evaluation.
"""
import torch
import torch.nn as nn
from transformers import AutoModel
import pandas as pd
from pathlib import Path

from model.base import BaseModel
from model.layers import AdditiveAttention, MultiHeadSelfAttention, DotProductClickPredictor
from model.utils import freeze_bert_layers


class NRMSBERT(BaseModel):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, model_config, dataset_config, device):
        super().__init__(model_config, dataset_config, device)

        self.news_encoder = self._NewsEncoder(model_config, self.plm, device)
        self.user_encoder = self._UserEncoder(model_config, dataset_config, self.plm, device)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
        }
        
        candidate_vecs_batched = self.news_encoder(batched_candidate)
        
        candidate_news_vector = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        num_clicked = len(clicked_news)
        
        # Concatenate all clicked news into a single batch
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
        }
        
        # Process all clicked news in a single forward pass
        clicked_vecs_batched = self.news_encoder(batched_clicked)
        
        clicked_vecs = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        
        # Apply mask
        if isinstance(clicked_news_mask, torch.Tensor):
            clicked_news_mask_tensor = clicked_news_mask
        else:
            clicked_news_mask_tensor = torch.stack(clicked_news_mask)

        if clicked_news_mask_tensor.dim() == 1:
            clicked_news_mask_tensor = clicked_news_mask_tensor.unsqueeze(0)
        elif clicked_news_mask_tensor.dim() == 2:
            if clicked_news_mask_tensor.size(0) == num_clicked and clicked_news_mask_tensor.size(1) == batch_size:
                clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)

        clicked_news_mask_tensor = clicked_news_mask_tensor.to(self.device)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_vecs = clicked_vecs * expanded_mask
        
        # Get user vector
        user_vector = self.user_encoder(clicked_vecs, mask=clicked_news_mask_tensor)
        
        # Compute click probability
        scale_factor = user_vector.size(-1) ** 0.5
        click_probability = self.click_predictor(candidate_news_vector, user_vector) / scale_factor
        
        return click_probability

    def get_news_vector(self, news):
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector, mask=None):
        return self.user_encoder(clicked_news_vector, mask=mask)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(news_vector, user_vector)

    class _NewsEncoder(nn.Module):
        def __init__(self, config, plm, device):
            super().__init__()
            self.config = config
            self.device = device
            self.plm = plm
            self.embedding_dim = self.plm.config.hidden_size
            self.multihead_self_attention = MultiHeadSelfAttention(self.embedding_dim, config['num_attention_heads'])
            self.additive_attention = AdditiveAttention(config['query_vector_dim'], self.embedding_dim)
            self.dropout = nn.Dropout(config.get('dropout_probability', 0.2))
            self.layer_norm = nn.LayerNorm(self.embedding_dim)

        def forward(self, news):
            input_ids = news["input_ids"].squeeze(1).to(self.device)
            word_mask = news["attention_mask"].squeeze(1).to(self.device)
            word_length = word_mask.sum(dim=1).to(self.device).long()
            
            news_vector = self.plm(input_ids=input_ids, attention_mask=word_mask).last_hidden_state
            
            residual = news_vector
            news_vector = self.multihead_self_attention(news_vector, length=word_length)
            news_vector = self.dropout(news_vector + residual)
            news_vector = self.layer_norm(news_vector)
            news_vector = self.additive_attention(news_vector, mask=word_mask)
            return news_vector

    class _UserEncoder(nn.Module):
        def __init__(self, config, dataset_config, plm, device):
            super().__init__()
            self.config = config
            self.device = device
            self.embedding_dim = plm.config.hidden_size
            self.multihead_self_attention = MultiHeadSelfAttention(self.embedding_dim, config['num_attention_heads'], device=device)
            self.additive_attention = AdditiveAttention(config['query_vector_dim'], self.embedding_dim)
            self.layer_norm = nn.LayerNorm(self.embedding_dim)
            self.dropout = nn.Dropout(config.get('dropout_probability', 0.2))

        def forward(self, user_vector, mask=None):
            x = user_vector
            residual = x
            if mask is not None:
                click_length = mask.sum(dim=1).to(self.device).long()
                click_length = click_length.clamp(min=1)
                x = self.multihead_self_attention(x, length=click_length)
            else:
                x = self.multihead_self_attention(x)
            
            x = x + residual
            x = self.layer_norm(x)
            x = self.dropout(x)
            final_user_vector = self.additive_attention(x, mask=mask)
            return final_user_vector


# ============================================================================
# NAML
# ============================================================================

class NAMLBERT(BaseModel):
    """
    NAML network.
    """
    def __init__(self, model_config, dataset_config, device):
        super().__init__(model_config, dataset_config, device)
        
        # Fallback configs
        if 'num_categories' not in self.model_config:
            self.model_config['num_categories'] = dataset_config.get('num_categories', 300)
        if 'num_subcategories' not in self.model_config:
            self.model_config['num_subcategories'] = dataset_config.get('num_subcategories', 1000)

        self.news_encoder = self._NewsEncoder(self.model_config, self.plm, device)
        self.user_encoder = self._UserEncoder(self.model_config, self.plm)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        batch_size = candidate_news[0]['input_ids'].shape[0]
        num_candidates = len(candidate_news)
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        
        num_clicked = len(clicked_news)
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        
        if isinstance(clicked_news_mask, torch.Tensor):
            clicked_news_mask_tensor = clicked_news_mask
        else:
            clicked_news_mask_tensor = torch.stack(clicked_news_mask)

        if clicked_news_mask_tensor.dim() == 1:
            clicked_news_mask_tensor = clicked_news_mask_tensor.unsqueeze(0)
        elif clicked_news_mask_tensor.dim() == 2:
            if clicked_news_mask_tensor.size(0) == num_clicked and clicked_news_mask_tensor.size(1) == batch_size:
                clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)

        clicked_news_mask_tensor = clicked_news_mask_tensor.to(self.device)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_news_vector = clicked_news_vector * expanded_mask
        
        user_vector = self.user_encoder(clicked_news_vector, mask=clicked_news_mask_tensor)
        click_probability = self.click_predictor(candidate_news_vector, user_vector)

        return click_probability

    def get_news_vector(self, news):
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector, mask=None):
        return self.user_encoder(clicked_news_vector, mask=mask)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(news_vector, user_vector)

    class _NewsEncoder(nn.Module):
        def __init__(self, config, plm, device):
            super().__init__()
            self.config = config
            self.device = device
            self.plm = plm
            self.embedding_dim = plm.config.hidden_size
            
            # CNN filters
            self.num_filters = config['num_filters']
            self.window_size = config['window_size']
            self.title_conv = nn.Conv1d(self.embedding_dim, self.num_filters, self.window_size, padding=1)
            self.abstract_conv = nn.Conv1d(self.embedding_dim, self.num_filters, self.window_size, padding=1)
            
            self.category_embedding = nn.Embedding(config['num_categories'] + 1, config['category_embedding_dim'], padding_idx=0)
            self.subcategory_embedding = nn.Embedding(config['num_subcategories'] + 1, config['category_embedding_dim'], padding_idx=0)
            self.category_dense = nn.Linear(config['category_embedding_dim'], self.num_filters)
            self.subcategory_dense = nn.Linear(config['category_embedding_dim'], self.num_filters)
            
            self.additive_attention = AdditiveAttention(config['query_vector_dim'], self.num_filters)

        def forward(self, news):
            # Title
            title_input_ids = news["input_ids"].squeeze(1).to(self.device)
            title_mask = news["attention_mask"].squeeze(1).to(self.device)
            title_emb = self.plm(input_ids=title_input_ids, attention_mask=title_mask).last_hidden_state
            title_emb = title_emb.transpose(1, 2) # [batch, emb, seq]
            title_conv = F.relu(self.title_conv(title_emb)).transpose(1, 2) # [batch, seq, filters]
            title_vec = self.additive_attention(title_conv, mask=title_mask)
            final_vec = title_vec
            
            # Category
            if 'category' in news:
                cat = news['category'].to(self.device)
                cat_emb = self.category_embedding(cat)
                cat_vec = F.relu(self.category_dense(cat_emb))
                # Combine (simple average or attention? Original NAML uses attention over views)
                # Simplified: just return title for now to match NRMS-like setup if views missing, 
                # but NAML is multi-view.
                # Let's stack views and attend.
                views = [title_vec, cat_vec]
                views_stack = torch.stack(views, dim=1) # [batch, 2, filters]
                final_vec = self.additive_attention(views_stack)

            return final_vec

    class _UserEncoder(nn.Module):
        def __init__(self, config, plm):
            super().__init__()
            self.additive_attention = AdditiveAttention(config['query_vector_dim'], config['num_filters'])

        def forward(self, user_vector, mask=None):
            return self.additive_attention(user_vector, mask=mask)


# ============================================================================
# LSTUR
# ============================================================================

class LSTURBERT(BaseModel):
    """
    LSTUR network.
    """
    def __init__(self, model_config, dataset_config, device):
        super().__init__(model_config, dataset_config, device)
        
        # Fallback for num_categories and num_subcategories
        if 'num_categories' not in self.model_config:
            self.model_config['num_categories'] = dataset_config.get('num_categories', 300)
        if 'num_subcategories' not in self.model_config:
            self.model_config['num_subcategories'] = dataset_config.get('num_subcategories', 1000)

        self.news_encoder = self._NewsEncoder(self.model_config, self.plm, device)
        self.user_encoder = self._UserEncoder(self.model_config, self.plm)
        self.click_predictor = DotProductClickPredictor()
        
        num_users = dataset_config.get('num_users', 1000000)
        self.user_embedding = nn.Embedding(num_users + 1, self.plm.config.hidden_size, padding_idx=0)

    def forward(self, candidate_news, clicked_news, clicked_news_mask, user):
        # Batch process candidate news
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
            'category': torch.cat([news['category'] for news in candidate_news], dim=0),
            'subcategory': torch.cat([news['subcategory'] for news in candidate_news], dim=0),
        }
        candidate_vecs_batched = self.news_encoder(batched_candidate)
        candidate_news_vector = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)

        # Batch process clicked news
        num_clicked = len(clicked_news)
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
            'category': torch.cat([news['category'] for news in clicked_news], dim=0),
            'subcategory': torch.cat([news['subcategory'] for news in clicked_news], dim=0),
        }
        clicked_vecs_batched = self.news_encoder(batched_clicked)
        clicked_news_vector = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        
        # Apply mask
        if isinstance(clicked_news_mask, torch.Tensor):
            clicked_news_mask_tensor = clicked_news_mask
        else:
            clicked_news_mask_tensor = torch.stack(clicked_news_mask)

        if clicked_news_mask_tensor.dim() == 1:
            clicked_news_mask_tensor = clicked_news_mask_tensor.unsqueeze(0)
        elif clicked_news_mask_tensor.dim() == 2:
            if clicked_news_mask_tensor.size(0) == num_clicked and clicked_news_mask_tensor.size(1) == batch_size:
                clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)

        clicked_news_mask_tensor = clicked_news_mask_tensor.to(self.device)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_news_vector = clicked_news_vector * expanded_mask

        user = self.user_embedding(user.to(self.device))

        user_vector = self.user_encoder(user, clicked_news_vector, mask=clicked_news_mask_tensor)

        click_probability = self.click_predictor(candidate_news_vector, user_vector)

        return click_probability

    def get_news_vector(self, news):
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_vector, mask=None):
        user = self.user_embedding(user.to(self.device))
        return self.user_encoder(user, clicked_news_vector, mask=mask)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(news_vector, user_vector)

    class _NewsEncoder(nn.Module):
        def __init__(self, config, plm, device):
            super().__init__()
            self.config = config
            self.device = device
            self.plm = plm
            self.embedding_dim = self.plm.config.hidden_size
            
            self.category_embedding = nn.Embedding(config['num_categories'] + 1, self.embedding_dim, padding_idx=0)
            self.subcategory_embedding = nn.Embedding(config['num_subcategories'] + 1, self.embedding_dim, padding_idx=0)
            
            self.additive_attention = AdditiveAttention(config['query_vector_dim'], self.embedding_dim)
            self.linear = nn.Linear(self.embedding_dim * 3, self.embedding_dim)

        def forward(self, news):
            text = {
                "input_ids": news["input_ids"].squeeze(1).to(self.device),
                "attention_mask": news["attention_mask"].squeeze(1).to(self.device)
            }
            news_vector = self.plm(**text).last_hidden_state
            
            # Apply dropout
            news_vector = F.dropout(news_vector, p=self.config.get('dropout_probability', 0.2), training=self.training)
            
            # Attended title
            title_vector = self.additive_attention(news_vector)
            
            # Category embeddings
            category_vector = self.category_embedding(news["category"].to(self.device))
            subcategory_vector = self.subcategory_embedding(news["subcategory"].to(self.device))
            
            # Concatenate
            combined_vector = torch.cat([title_vector, category_vector, subcategory_vector], dim=1)
            combined_vector = self.linear(combined_vector)
            
            return combined_vector

    class _UserEncoder(nn.Module):
        def __init__(self, config, plm):
            super().__init__()
            self.embedding_dim = plm.config.hidden_size
            self.gru = nn.GRU(
                input_size=self.embedding_dim,
                hidden_size=self.embedding_dim,
                batch_first=True
            )
            
        def forward(self, user, clicked_news_vector, mask=None):
            initial_state = user.unsqueeze(0)
            
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()
                lengths = lengths.clamp(min=1)
                from torch.nn.utils.rnn import pack_padded_sequence
                packed_input = pack_padded_sequence(
                    clicked_news_vector, 
                    lengths, 
                    batch_first=True, 
                    enforce_sorted=False
                )
                _, final_state = self.gru(packed_input, initial_state)
            else:
                _, final_state = self.gru(clicked_news_vector, initial_state)
            
            return final_state.squeeze(0)


# ============================================================================
# NPA
# ============================================================================

from model.layers import SimilarityAttention

class NPABERT(BaseModel):
    """
    NPA Network
    """
    def __init__(self, model_config, dataset_config, device):
        super().__init__(model_config, dataset_config, device)
        
        self.news_encoder = self._NewsEncoder(model_config, self.plm, device)
        self.user_encoder = self._UserEncoder(model_config, self.plm)
        self.click_predictor = DotProductClickPredictor()
        
        # User embedding
        num_users = dataset_config.get('num_users', 1000000)
        # Check mapping file logic from original if needed, but safe default usually works
        mapping_path = Path(dataset_config["path"]).joinpath("train", "user2int.tsv")
        if mapping_path.exists():
            try:
                mapping_df = pd.read_table(mapping_path, header=None, names=["user", "int"])
                inferred_num_users = int(mapping_df["int"].max())
                num_users = inferred_num_users
            except:
                pass
        
        self.user_embedding = nn.Embedding(num_users + 1, self.plm.config.hidden_size, padding_idx=0)

    def forward(self, candidate_news, clicked_news, clicked_news_mask, user):
        # Batch process candidate news
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
        }
        
        user_vec = self.user_embedding(user.to(self.device))
        expanded_user_vec = user_vec.repeat_interleave(num_candidates, dim=0)
        
        candidate_vecs_batched = self.news_encoder(batched_candidate, expanded_user_vec)
        candidate_news_vector = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        
        # Batch process clicked news
        num_clicked = len(clicked_news)
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
        }
        
        expanded_clicked_user_vec = user_vec.repeat_interleave(num_clicked, dim=0)
        clicked_vecs_batched = self.news_encoder(batched_clicked, expanded_clicked_user_vec)
        clicked_vecs = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)

        # Apply mask
        if isinstance(clicked_news_mask, torch.Tensor):
            clicked_news_mask_tensor = clicked_news_mask
        else:
            clicked_news_mask_tensor = torch.stack(clicked_news_mask)
            
        if clicked_news_mask_tensor.dim() == 1:
            clicked_news_mask_tensor = clicked_news_mask_tensor.unsqueeze(0)
        elif clicked_news_mask_tensor.dim() == 2:
            if clicked_news_mask_tensor.size(0) == num_clicked and clicked_news_mask_tensor.size(1) == batch_size:
                clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)

        clicked_news_mask_tensor = clicked_news_mask_tensor.to(self.device)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_vecs = clicked_vecs * expanded_mask
        
        user_vector = self.user_encoder(user_vec, clicked_vecs)
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        
        return click_probability

    def get_news_vector(self, news):
        # Fallback non-personalized encoding
        text = {
            "input_ids": news["input_ids"].squeeze(1).to(self.device),
            "attention_mask": news["attention_mask"].squeeze(1).to(self.device),
        }
        return self.plm(**text).last_hidden_state[:, 0, :]

    def get_user_vector(self, user, clicked_news_vector, mask=None):
        user_vec = self.user_embedding(user.to(self.device))
        return self.user_encoder(user_vec, clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(news_vector, user_vector)

    class _NewsEncoder(nn.Module):
        def __init__(self, config, plm, device):
            super().__init__()
            self.config = config
            self.plm = plm
            self.device = device
            self.personalized_attention = SimilarityAttention()
            
        def forward(self, news, user_vector):
            text = {
                "input_ids": news["input_ids"].squeeze(1).to(self.device),
                "attention_mask": news["attention_mask"].squeeze(1).to(self.device),
            }
            token_embeddings = self.plm(**text).last_hidden_state
            token_embeddings = F.dropout(
                token_embeddings, 
                p=float(self.config.get("dropout_probability", 0.0)),
                training=self.training
            )
            # user_vector: [batch, dim], token_embeddings: [batch, seq, dim]
            return self.personalized_attention(user_vector, token_embeddings)

    class _UserEncoder(nn.Module):
         def __init__(self, config, plm):
            super().__init__()
            self.personalized_attention = SimilarityAttention()
            
         def forward(self, user_vector, clicked_news_vector):
             # user_vector: [batch, dim]
             # clicked_news_vector: [batch, history_len, dim]
             return self.personalized_attention(user_vector, clicked_news_vector)


