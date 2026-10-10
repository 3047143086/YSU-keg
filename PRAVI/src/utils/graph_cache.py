# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import pickle
from logging import getLogger


class GraphCacheManager:
    """Graph cache manager"""
    
    def __init__(self, data_path: str, dataset_name: str, cache_dir: str = 'cache'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.cache_dir = os.path.join(data_path, dataset_name, cache_dir)
        self.logger = getLogger()
        
        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.logger.info(f'Graph cache directory: {self.cache_dir}')
    
    def get_model_cache_dir(self, model_name: str) -> str:
        """Get cache directory for specific model"""
        model_cache_dir = os.path.join(self.cache_dir, model_name)
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir)
        return model_cache_dir
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cache
        
        Args:
            model_name: If specified, only clear cache for this model; otherwise clear all
        """
        if model_name:
            model_cache_dir = self.get_model_cache_dir(model_name)
            if os.path.exists(model_cache_dir):
                import shutil
                shutil.rmtree(model_cache_dir)
                self.logger.info(f'Cleared cache for model: {model_name}')
        else:
            if os.path.exists(self.cache_dir):
                import shutil
                shutil.rmtree(self.cache_dir)
                self.logger.info('Cleared all cache')
    
    def save_graph(self, model_name: str, graph_name: str, graph_data, metadata: Dict = None):
        """Save graph structure to cache
        
        Args:
            model_name: Model name
            graph_name: Graph name
            graph_data: Graph data (torch.Tensor, scipy.sparse, dict, etc.)
            metadata: Metadata (configuration parameters, etc.)
        """
        model_cache_dir = self.get_model_cache_dir(model_name)
        save_path = os.path.join(model_cache_dir, f'{graph_name}.pt')
        
        # Save graph data and metadata
        save_dict = {
            'graph': graph_data,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f'Saved graph {graph_name} to {save_path}')
    
    def load_graph(self, model_name: str, graph_name: str) -> Tuple:
        """Load graph structure from cache
        
        Args:
            model_name: Model name
            graph_name: Graph name
            
        Returns:
            (graph_data, metadata)
        """
        model_cache_dir = self.get_model_cache_dir(model_name)
        load_path = os.path.join(model_cache_dir, f'{graph_name}.pt')
        
        if os.path.exists(load_path):
            load_dict = torch.load(load_path)
            self.logger.info(f'Loaded graph {graph_name} from {load_path}')
            return load_dict.get('graph'), load_dict.get('metadata', {})
        else:
            self.logger.info(f'Graph {graph_name} not found in cache')
            return None, None
    
    def has_cache(self, model_name: str, graph_name: str) -> bool:
        """Check if cache exists"""
        model_cache_dir = self.get_model_cache_dir(model_name)
        load_path = os.path.join(model_cache_dir, f'{graph_name}.pt')
        return os.path.exists(load_path)


class DualGNNPreprocessor:
    """DualGNN preprocessing: build user-user graph"""
    
    def __init__(self, cache_manager: GraphCacheManager):
        self.cache_manager = cache_manager
        self.logger = getLogger()
    
    def build_user_user_graph(self, interaction_matrix: sp.coo_matrix, 
                             user_features: Optional[torch.Tensor] = None,
                             k: int = 40, 
                             construction: str = 'weighted_sum') -> Dict:
        """Build user-user KNN graph
        
        Args:
            interaction_matrix: User-item interaction matrix
            user_features: User features (optional)
            k: K value for KNN
            construction: Construction method ('weighted_sum', 'knn', etc.)
            
        Returns:
            user_graph_dict: User graph dictionary {user_id: [neighbor_user_ids]}
        """
        self.logger.info(f'Building user-user graph with k={k}, construction={construction}')
        
        n_users = interaction_matrix.shape[0]
        
        # Build user-user similarity based on interactions
        # Using cosine similarity
        user_item_matrix = interaction_matrix.tocsr()
        
        # Compute user-user cosine similarity
        user_norms = sp.linalg.norm(user_item_matrix, axis=1)
        user_norms[user_norms == 0] = 1  # Avoid division by zero
        
        # Normalize
        user_item_normalized = user_item_matrix.multiply(1.0 / user_norms.reshape(-1, 1))
        
        # Compute similarity matrix
        sim_matrix = user_item_normalized.dot(user_item_normalized.T)
        
        # Convert to dense matrix for top-k operation
        sim_matrix = sim_matrix.toarray()
        
        # Find k most similar users for each user
        user_graph_dict = {}
        for u in range(n_users):
            sim_scores = sim_matrix[u]
            sim_scores[u] = -np.inf  # Exclude self
            
            top_k_neighbors = np.argsort(sim_scores)[-k:]
            user_graph_dict[u] = top_k_neighbors.tolist()
        
        self.logger.info(f'Built user-user graph with {len(user_graph_dict)} users')
        
        return user_graph_dict
    
    def save_user_graph(self, user_graph_dict: Dict, k: int, construction: str):
        """Save user-user graph to cache"""
        metadata = {
            'k': k,
            'construction': construction,
            'n_users': len(user_graph_dict)
        }
        
        self.cache_manager.save_graph(
            model_name='DualGNN',
            graph_name='user_graph_dict',
            graph_data=user_graph_dict,
            metadata=metadata
        )
    
    def load_user_graph(self, k: int, construction: str) -> Optional[Dict]:
        """Load user-user graph from cache"""
        # Check if matching cache exists
        graph_data, metadata = self.cache_manager.load_graph('DualGNN', 'user_graph_dict')
        
        if graph_data is not None:
            # Verify if parameters match
            if metadata.get('k') == k and metadata.get('construction') == construction:
                return graph_data
            else:
                self.logger.info('Cache parameters mismatch, rebuilding...')
        
        return None


class FREEDOMPreprocessor:
    """FREEDOM preprocessing: build item-item KNN graph"""
    
    def __init__(self, cache_manager: GraphCacheManager):
        self.cache_manager = cache_manager
        self.logger = getLogger()
    
    def build_item_item_knn_graph(self, item_features: torch.Tensor, 
                                  knn_k: int = 10,
                                  mm_image_weight: float = 0.5,
                                  text_features: Optional[torch.Tensor] = None) -> torch.sparse.FloatTensor:
        """Build item-item KNN graph
        
        Args:
            item_features: Item features (visual)
            knn_k: K value for KNN
            mm_image_weight: Weight for visual and text features
            text_features: Text features (optional)
            
        Returns:
            mm_adj: Multimodal adjacency matrix (sparse tensor)
        """
        self.logger.info(f'Building item-item KNN graph with knn_k={knn_k}')
        
        device = item_features.device
        
        # Normalize features
        context_norm = item_features.div(torch.norm(item_features, p=2, dim=-1, keepdim=True))
        
        # Compute similarity matrix
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        
        # Get k nearest neighbors
        _, knn_ind = torch.topk(sim, knn_k, dim=-1)
        
        adj_size = sim.size()
        del sim
        
        # Build sparse adjacency matrix
        indices0 = torch.arange(knn_ind.shape[0]).to(device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        
        # Normalized Laplacian
        mm_adj = self.compute_normalized_laplacian(indices, adj_size)
        
        # If text features exist, fuse them
        if text_features is not None:
            self.logger.info('Fusing text features...')
            text_norm = text_features.div(torch.norm(text_features, p=2, dim=-1, keepdim=True))
            text_sim = torch.mm(text_norm, text_norm.transpose(1, 0))
            _, text_knn_ind = torch.topk(text_sim, knn_k, dim=-1)
            
            text_indices = torch.stack((
                torch.flatten(torch.arange(knn_k).to(device).unsqueeze(0).expand(-1, knn_k)),
                torch.flatten(text_knn_ind)
            ), 0)
            
            text_adj = self.compute_normalized_laplacian(text_indices, adj_size)
            
            # Weighted fusion
            mm_adj = mm_image_weight * mm_adj + (1.0 - mm_image_weight) * text_adj
        
        self.logger.info(f'Built item-item KNN graph with shape {mm_adj.shape}')
        
        return mm_adj
    
    def compute_normalized_laplacian(self, indices: torch.Tensor, adj_size: torch.Size) -> torch.sparse.FloatTensor:
        """Compute normalized Laplacian matrix"""
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def save_mm_adj(self, mm_adj: torch.sparse.FloatTensor, knn_k: int, mm_image_weight: float):
        """Save multimodal adjacency matrix to cache"""
        metadata = {
            'knn_k': knn_k,
            'mm_image_weight': mm_image_weight,
            'shape': list(mm_adj.shape)
        }
        
        self.cache_manager.save_graph(
            model_name='FREEDOM',
            graph_name=f'mm_adj_freedomdsp_{knn_k}_{int(10*mm_image_weight)}',
            graph_data=mm_adj,
            metadata=metadata
        )
    
    def load_mm_adj(self, knn_k: int, mm_image_weight: float) -> Optional[torch.sparse.FloatTensor]:
        """Load multimodal adjacency matrix from cache"""
        graph_name = f'mm_adj_freedomdsp_{knn_k}_{int(10*mm_image_weight)}'
        graph_data, metadata = self.cache_manager.load_graph('FREEDOM', graph_name)
        
        if graph_data is not None:
            # Verify parameters
            if metadata.get('knn_k') == knn_k and metadata.get('mm_image_weight') == mm_image_weight:
                return graph_data
        
        return None