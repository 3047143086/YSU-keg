# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import os
import glob
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    """Base class for all models"""
    
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        """Print model parameters"""
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """General recommender base class - automatic modality feature discovery"""
    
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # Basic information
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # Training configuration
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # Automatically discover and load all modality features
        self._load_all_modalities(config)
    
    def _load_all_modalities(self, config):
        """
        Automatically scan all *_feat.npy files in the data directory
        and load them as corresponding attributes: self.v_feat, self.t_feat, self.audio_feat, etc.
        """
        if config.get('end2end', False) or not config.get('is_multimodal_model', True):
            return
        
        dataset_path = os.path.abspath(os.path.join(config['data_path'], config['dataset']))
        
        if not os.path.exists(dataset_path):
            print(f"[Warning] Dataset path not found: {dataset_path}")
            return
        
        # Scan all *_feat.npy and *_feat.pt files
        feat_patterns = [
            os.path.join(dataset_path, '*_feat.npy'),
            os.path.join(dataset_path, '*_feat.pt')
        ]
        
        feat_files = []
        for pattern in feat_patterns:
            feat_files.extend(glob.glob(pattern))
        
        if not feat_files:
            print(f"[Warning] No feature files found in {dataset_path}")
            return
        
        print(f"\n[Auto-Discovery] Found {len(feat_files)} modality feature files:")
        
        # Load each feature file
        for feat_file in feat_files:
            # Extract modality name from filename
            filename = os.path.basename(feat_file)
            modality_name = filename.replace('_feat.npy', '').replace('_feat.pt', '')
            
            # Convert to attribute name (e.g., image -> v_feat, text -> t_feat)
            attr_name = self._get_modality_attribute_name(modality_name)
            
            try:
                # Load feature
                if feat_file.endswith('.npy'):
                    feat_tensor = torch.from_numpy(np.load(feat_file, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
                elif feat_file.endswith('.pt'):
                    feat_tensor = torch.load(feat_file, map_location=self.device)
                else:
                    continue
                
                # Set as attribute
                setattr(self, attr_name, feat_tensor)
                print(f"  ✓ {modality_name} -> {attr_name} (shape: {feat_tensor.shape})")
                
            except Exception as e:
                print(f"  ✗ Failed to load {filename}: {e}")
        
        print()
    
    def _get_modality_attribute_name(self, modality_name: str) -> str:
        """
        Convert modality name to attribute name
        
        Common mappings:
        - visual, image, vision, v, img -> visual_feat
        - textual, text, t, txt -> textual_feat
        - audio, a, sound, acoustic -> audio_feat
        - gpt, llm -> gpt_feat
        - caption, cap -> caption_feat
        - knowledge, k, kg -> knowledge_feat
        - others -> {modality_name}_feat
        """
        modality_lower = modality_name.lower()
        
        # Common modality mappings - using full names
        if modality_lower in ['visual', 'image', 'vision', 'v', 'img']:
            return 'visual_feat'
        elif modality_lower in ['textual', 'text', 't', 'txt']:
            return 'textual_feat'
        elif modality_lower in ['audio', 'a', 'sound', 'acoustic']:
            return 'audio_feat'
        elif modality_lower in ['gpt', 'llm']:
            return 'gpt_feat'
        elif modality_lower in ['caption', 'cap']:
            return 'caption_feat'
        elif modality_lower in ['knowledge', 'k', 'kg']:
            return 'knowledge_feat'
        else:
            # Use original name for other modalities
            return f'{modality_lower}_feat'
    
    def get_available_modalities(self):
        """Get all available modalities"""
        modalities = []
        for attr in dir(self):
            if attr.endswith('_feat') and getattr(self, attr) is not None:
                modalities.append(attr.replace('_feat', ''))
        return modalities