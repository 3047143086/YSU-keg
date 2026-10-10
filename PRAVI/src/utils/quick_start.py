# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # Merge configuration
    config = Config(model, dataset, config_dict)
    
    init_logger(config)
    logger = getLogger()
    
    # Print configuration
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # Load data
    dataset_obj = RecDataset(config)
    logger.info(str(dataset_obj))

    train_dataset, valid_dataset, test_dataset = dataset_obj.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # Wrap as data loaders
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])

    # Hyperparameter search
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0
    
    # Global best results (across all hyperparameter combinations)
    global_best_valid_result = None
    global_best_test_result = None
    global_best_params = None
    global_best_score = -1

    logger.info('\n\n=================================\n\n')

    # Hyperparameter combinations
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    
    for param in config['hyper_parameters']:
        hyper_ls.append(config[param] or [None])
    
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    
    for hyper_tuple in combinators:
        # Set random seed
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info(f'\n{"="*70}')
        logger.info(f'========={idx+1}/{total_loops}: Parameters: {config["hyper_parameters"]}={hyper_tuple}')
        logger.info(f'{"="*70}\n')

        # Data loader preprocessing
        train_data.pretrain_setup()
        
        # Load model
        model_instance = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model_instance)

        # Load trainer
        trainer = get_trainer()(config, model_instance)
        
        # Train
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # Update best results for current hyperparameter combination
        if best_valid_score > best_test_value:
            best_test_value = best_valid_score
            best_test_idx = idx
        
        # Update global best results (across all hyperparameters)
        if best_valid_score > global_best_score:
            global_best_score = best_valid_score
            global_best_valid_result = best_valid_result
            global_best_test_result = best_test_upon_valid
            global_best_params = hyper_tuple
            
            # Update global best in visualizer
            trainer.visualizer.update_global_best(best_valid_result, best_test_upon_valid, hyper_tuple)
        
        idx += 1

        logger.info(f'\n📊 This run - Valid: {dict2str(best_valid_result)}')
        logger.info(f'📊 This run - Test: {dict2str(best_test_upon_valid)}')
        logger.info(f'\n🏆 Global BEST (across all hyper-parameters):')
        logger.info(f'   Parameters: {config["hyper_parameters"]}={global_best_params}')
        logger.info(f'   Valid: {dict2str(global_best_valid_result)}')
        logger.info(f'   Test: {dict2str(global_best_test_result)}\n')

    # Final results
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(
            config['hyper_parameters'], p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ GLOBAL BEST ████████████████')
    logger.info(f'🏆 Best Hyper-parameters: {config["hyper_parameters"]}={global_best_params}')
    logger.info(f'📊 Valid Results: {dict2str(global_best_valid_result)}')
    logger.info(f'📊 Test Results: {dict2str(global_best_test_result)}')
    logger.info('██████████████████████████████████████████\n')
