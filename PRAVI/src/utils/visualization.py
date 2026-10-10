# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import os
import numpy as np
from typing import Dict, List, Optional


plt = None


class TrainingVisualizer:
    """Training process visualizer - tracks global best results across hyperparameter search"""
    
    def __init__(self, config, save_dir: str = None):
        global plt
        self.config = config
        self.enabled = config.get('enable_visualization', False)
        if self.enabled and plt is None:
            import matplotlib.pyplot as pyplot

            plt = pyplot
        
        # Save to log/{model}/{dataset}/visualization/
        if save_dir is None:
            log_dir = os.path.join('./log', config['model'], config['dataset'])
            self.save_dir = os.path.join(log_dir, 'visualization')
        else:
            self.save_dir = save_dir
        
        # Create save directory
        if self.enabled and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Store training history for current run
        self.history = {
            'train_loss': [],
            'valid_metrics': {},
            'test_metrics': {}
        }
        
        # Best results for current training run
        self.best_valid_score = -1
        self.best_epoch = -1
        self.best_valid_result = None
        self.best_test_result = None
        
        # Global best results across all hyperparameter configurations
        self.global_best_valid_score = -1
        self.global_best_valid_result = None
        self.global_best_test_result = None
        self.global_best_params = None
        
        # Configuration
        self.plot_style = config.get('plot_style', 'seaborn-v0_8')
        self.dpi = config.get('plot_dpi', 150)
        self.fig_size = config.get('plot_figsize', (10, 6))
        
    def record_loss(self, epoch: int, loss: float):
        """Record training loss"""
        if not self.enabled:
            return
            
        self.history['train_loss'].append({
            'epoch': epoch,
            'loss': loss
        })
    
    def record_metrics(self, epoch: int, valid_result: Dict, test_result: Optional[Dict] = None):
        """Record evaluation metrics and update best results for current training run"""
        if not self.enabled:
            return
        
        # Get validation score
        valid_metric = self.config.get('valid_metric', 'Recall@20').lower()
        valid_score = valid_result.get(valid_metric, 0.0)
        
        # Record results for current epoch
        self.history['valid_metrics'][epoch] = {
            'valid': valid_result,
            'test': test_result,
            'score': valid_score
        }
        
        # Update best results for current training run
        valid_metric_bigger = self.config.get('valid_metric_bigger', True)
        
        is_better = False
        if valid_metric_bigger:
            is_better = valid_score > self.best_valid_score
        else:
            is_better = valid_score < self.best_valid_score
        
        if is_better:
            self.best_valid_score = valid_score
            self.best_epoch = epoch
            self.best_valid_result = valid_result
            self.best_test_result = test_result
            
            print(f"\n🏆 New best at epoch {epoch + 1}: {valid_metric} = {valid_score:.4f}")
    
    def update_global_best(self, valid_result: Dict, test_result: Optional[Dict], 
                          hyper_params: tuple):
        """Update global best results across hyperparameter search"""
        if not self.enabled:
            return
        
        valid_metric = self.config.get('valid_metric', 'Recall@20').lower()
        valid_score = valid_result.get(valid_metric, 0.0)
        
        # Update global best
        valid_metric_bigger = self.config.get('valid_metric_bigger', True)
        
        is_better = False
        if valid_metric_bigger:
            is_better = valid_score > self.global_best_valid_score
        else:
            is_better = valid_score < self.global_best_valid_score
        
        if is_better:
            self.global_best_valid_score = valid_score
            self.global_best_valid_result = valid_result
            self.global_best_test_result = test_result
            self.global_best_params = hyper_params
            
            # Save visualizations for global best results
            self._save_global_best_plots(valid_result, test_result, hyper_params)
    
    def _save_best_plots(self, epoch: int, valid_result: Dict, test_result: Optional[Dict]):
        """Save charts for best results of current training run (intermediate process only)"""
        pass  # No longer save individual run best, only save global best
    
    def _save_global_best_plots(self, valid_result: Dict, test_result: Optional[Dict], 
                               hyper_params: tuple):
        """Save charts for global best results across hyperparameter search (overwrites old files)"""
        if not self.enabled:
            return
        
        # Base filename
        base_filename = f'{self.config["model"]}_{self.config["dataset"]}'
        
        # 1. Save global best loss curve (using history from current training run)
        self._plot_loss_curve_global(f'{base_filename}_loss_best.png')
        
        # 2. Save global best metrics curve
        self._plot_metrics_curve_global(f'{base_filename}_metrics_best.png', valid_result, test_result)
        
        # 3. Save global best results summary
        self._save_global_best_summary(valid_result, test_result, hyper_params)
    
    def _plot_loss_curve_global(self, filename: str):
        """Plot and save global best loss curve (complete history for current hyperparameter)"""
        if len(self.history['train_loss']) == 0:
            return
        
        epochs = [item['epoch'] for item in self.history['train_loss']]
        losses = [item['loss'] for item in self.history['train_loss']]
        
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        plt.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=3)
        
        # Mark best epoch
        if self.best_epoch >= 0 and self.best_epoch < len(losses):
            plt.axvline(x=self.best_epoch, color='r', linestyle='--', linewidth=2, 
                       label=f'Best Epoch {self.best_epoch + 1}')
            plt.plot(self.best_epoch, losses[self.best_epoch], 'ro', markersize=10)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title(f'{self.config["model"]} on {self.config["dataset"]}', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_curve_global(self, filename: str, valid_result: Dict, test_result: Optional[Dict]):
        """Plot and save global best metrics curve"""
        if len(self.history['valid_metrics']) == 0:
            return
        
        # Prepare data
        epochs = []
        valid_scores = []
        test_scores = []
        
        for epoch, data in sorted(self.history['valid_metrics'].items()):
            epochs.append(epoch)
            valid_scores.append(data['score'])
            test_scores.append(data['test'].get(self.config.get('valid_metric', 'Recall@20').lower(), 0.0) if data['test'] else 0.0)
        
        # Plot curves
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        ax.plot(epochs, valid_scores, 'b-', linewidth=2, marker='s', markersize=4, label='Valid')
        ax.plot(epochs, test_scores, 'r-', linewidth=2, marker='o', markersize=4, label='Test')
        
        # Mark best point
        if self.best_epoch >= 0:
            ax.axvline(x=self.best_epoch, color='g', linestyle='--', linewidth=2, 
                      label=f'Best Epoch {self.best_epoch + 1}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Metric Score', fontsize=12)
        ax.set_title(f'{self.config["model"]} on {self.config["dataset"]}', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _save_global_best_summary(self, valid_result: Dict, test_result: Optional[Dict], 
                                 hyper_params: tuple):
        """Save summary of global best results across hyperparameter search"""
        import json
        
        summary = {
            'model': self.config['model'],
            'dataset': self.config['dataset'],
            'best_hyper_parameters': dict(zip(self.config.get('hyper_parameters', []), hyper_params)),
            'best_epoch': self.best_epoch + 1,
            'valid_metric': self.config.get('valid_metric', 'Recall@20'),
            'best_valid_score': self.global_best_valid_score,
            'best_valid_result': valid_result,
            'best_test_result': test_result
        }
        
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        serializable_summary = {
            k: convert_to_serializable(v) for k, v in summary.items()
        }
        
        # Save as JSON (overwrite old file, keep only global best)
        save_path = os.path.join(self.save_dir, f'{self.config["model"]}_{self.config["dataset"]}_best_summary.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
    
    def plot_all(self):
        """Generate best results visualization charts"""
        if not self.enabled:
            return
        
        print("\n📊 Generating best training visualization...")
        print(f"🏆 Best results at epoch {self.best_epoch + 1}")
        print(f"📁 Location: {self.save_dir}")
        print("✅ Visualization completed!\n")
