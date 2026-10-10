# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from time import time
from logging import getLogger
from typing import Dict, Tuple, Optional

from utils.utils import early_stopping, dict2str


class MRSTrainer:
    """
    1. More concise training loop
    2. Better logging
    3. Early stopping support
    """
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()
        
        # Training configuration
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        
        # Evaluation configuration
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.device = config['device']
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Learning rate scheduler
        lr_scheduler = config.get('learning_rate_scheduler', [1.0, 50])
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        
        # Early stopping configuration
        self.best_valid_score = -1
        self.cur_step = 0
        self.best_valid_result = {}
        self.best_test_upon_valid = {}
        
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if self.learner.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.learner.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.learner.lower() == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.learner.lower() == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            self.logger.warning(f"Unknown optimizer: {self.learner}, using Adam")
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _train_epoch(self, train_data, epoch_idx: int) -> float:
        """
        Train for one epoch
        
        Args:
            train_data: Training data loader
            epoch_idx: Current epoch index
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, interactions in enumerate(train_data):
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.model.calculate_loss(interactions)
            
            # Support multiple losses
            if isinstance(losses, tuple):
                loss = sum(losses)
            else:
                loss = losses
            
            # Check for NaN
            if torch.isnan(loss):
                self.logger.warning(f"Loss is NaN at epoch {epoch_idx}, batch {batch_idx}")
                return float('nan')
            
            # Backward propagation
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Update learning rate
        self.lr_scheduler.step()
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    @torch.no_grad()
    def _evaluate(self, eval_data) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        # Should call evaluator here, simplified for now
        # Actual implementation requires complete evaluation logic
        return {}
    
    def fit(self, train_data, valid_data=None, test_data=None, verbose=True) -> Tuple[float, Dict, Dict]:
        """
        Train the model
        
        Args:
            train_data: Training data
            valid_data: Validation data
            test_data: Test data
            verbose: Whether to print logs
            
        Returns:
            (best_valid_score, best_valid_result, best_test_upon_valid)
        """
        for epoch_idx in range(self.epochs):
            # Training
            training_start = time()
            self.model.pre_epoch_processing() if hasattr(self.model, 'pre_epoch_processing') else None
            
            train_loss = self._train_epoch(train_data, epoch_idx)
            
            if torch.isnan(torch.tensor(train_loss)):
                self.logger.error("Training stopped due to NaN loss")
                break
            
            training_end = time()
            
            if verbose:
                self.logger.info(f"Epoch {epoch_idx}: train_loss={train_loss:.4f}, "
                               f"time={training_end - training_start:.2f}s")
            
            # Evaluation
            if (epoch_idx + 1) % self.eval_step == 0 and valid_data is not None:
                valid_start = time()
                valid_score, valid_result = self._evaluate(valid_data)
                valid_end = time()
                
                # Early stopping check
                _, _, stop_flag, update_flag = early_stopping(
                    valid_score, 
                    self.best_valid_score, 
                    self.cur_step,
                    max_step=self.stopping_step, 
                    bigger=self.valid_metric_bigger
                )
                
                if update_flag:
                    self.best_valid_score = valid_score
                    self.best_valid_result = valid_result
                    if test_data is not None:
                        _, self.best_test_upon_valid = self._evaluate(test_data)
                
                if verbose:
                    self.logger.info(f"Valid score: {valid_score:.4f}, time: {valid_end - valid_start:.2f}s")
                
                if stop_flag:
                    self.logger.info(f"Early stopping at epoch {epoch_idx}")
                    break
                
                self.cur_step = 0 if update_flag else self.cur_step + 1
        
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid