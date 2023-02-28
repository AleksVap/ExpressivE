import json
import os
import sys

from pykeen.models import BoxE, RotatE
from pykeen.pipeline import pipeline
from torch.optim import Adam, Adagrad

from ExpressivEModel import ExpressivE
from Analysis_Utils import analyze_checkpoints


def parse_kwargs(**kwargs):
    if 'config' in kwargs.keys():
        config_path = kwargs['config']

        with open(config_path, 'r') as f:
            config = json.loads(f.read())
    else:
        raise Exception('No config input file!')

    if 'gpu' in kwargs.keys():
        gpu = kwargs['gpu']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    if 'seeds' in kwargs.keys():
        seeds = kwargs['seeds'].split(',')
        seeds = [int(i) for i in seeds]
    else:
        seeds = [1, 2, 3]

    if 'expName' in kwargs.keys():
        sub_dir = kwargs['expName']

    if 'test' in kwargs.keys():
        if kwargs['test'] == 'true':
            test = True
        elif kwargs['test'] == 'false':
            test = False
        else:
            raise Exception(
                'Invalid value %s for parameter test. Parameter test may only be <true> or <false>.' % kwargs[
                    'test'])
    else:
        raise Exception(
            'Parameter <test> needs to be specified.')

    if 'train' in kwargs.keys():
        if kwargs['train'] == 'true':
            train = True
        elif kwargs['train'] == 'false':
            train = False
        else:
            raise Exception(
                'Invalid value %s for parameter train. Parameter train may only be <true> or <false>.' % kwargs[
                    'train'])
    else:
        raise Exception(
            'Parameter <train> needs to be specified.')

    return config_path, config, sub_dir, seeds, train, test


def parse_config(config):
    if config['optimizer'] == 'Adam':
        optimizer = Adam
    elif config['optimizer'] == 'Adagrad':
        optimizer = Adagrad
    else:
        raise Exception('Optimizer %s unknown!' % config['optimizer'])

    if config['model'] == 'ExpressivEModel':
        model = ExpressivE
        model_kwargs = dict(embedding_dim=config['model_kwargs']['embedding_dim'], p=config['model_kwargs']['p'],
                            min_denom=config['model_kwargs']['min_denom'], tanh_map=config['model_kwargs']['tanh_map'])

        if 'interactionMode' in config['model_kwargs']:
            model_kwargs['interactionMode'] = config['model_kwargs']['interactionMode']

    elif config['model'] == 'BoxE':
        model = BoxE
        model_kwargs = dict(embedding_dim=config['model_kwargs']['embedding_dim'], p=config['model_kwargs']['p'])

    elif config['model'] == 'RotatE':
        model = RotatE
        model_kwargs = dict(embedding_dim=config['model_kwargs']['embedding_dim'])

    else:
        raise Exception('Model %s unknown!' % config['model'])

    return optimizer, model, model_kwargs


def evaluate_and_save_final_result(seeds, res, experiment_dir, config_path, sub_dir):
    for seed in seeds:
        sub_dir_seed = '/' + sub_dir + '_seed_' + str(seed)
        res[seed] = analyze_checkpoints(experiment_dir, config_path, sub_dir_seed)

    final_res = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@10': 0}

    for seed in seeds:
        final_res['MRR'] = final_res['MRR'] + res[seed]['test_res']['both.realistic.inverse_harmonic_mean_rank']
        final_res['Hits@1'] = final_res['Hits@1'] + res[seed]['test_res']['both.realistic.hits_at_1']
        final_res['Hits@3'] = final_res['Hits@3'] + res[seed]['test_res']['both.realistic.hits_at_3']
        final_res['Hits@10'] = final_res['Hits@10'] + res[seed]['test_res']['both.realistic.hits_at_10']

    for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
        final_res[metric] = round(final_res[metric] / len(seeds), 3)

    final_res_dir = experiment_dir + '/final_result' + '/' + sub_dir
    file_name = 'final_res_seeds_%s.json' % ('_'.join(map(str, seeds)))
    final_res_path = os.path.join(final_res_dir, file_name)

    if not os.path.exists(final_res_dir):
        os.makedirs(final_res_dir)
    with open(final_res_path, 'w') as fp:
        json.dump(final_res, fp)


def main(**kwargs):
    config_path, config, sub_dir, seeds, train, test = parse_kwargs(**kwargs)
    optimizer, model, model_kwargs = parse_config(config)

    res = {}

    experiment_dir = './Benchmarking'

    if train:
        for seed in seeds:
            sub_dir_seed = '/' + sub_dir + '_seed_' + str(seed)

            pipeline(
                model=model,
                model_kwargs=model_kwargs,
                dataset=config['dataset'],
                dataset_kwargs=dict(create_inverse_triples=config['dataset_kwargs']['create_inverse_triples']),
                optimizer=optimizer,
                optimizer_kwargs=dict(lr=config['optimizer_kwargs']['lr']),
                loss=config['loss'],
                loss_kwargs=dict(reduction=config['loss_kwargs']['reduction'],
                                 adversarial_temperature=config['loss_kwargs']['adversarial_temperature'],
                                 margin=config['loss_kwargs']['margin']),
                result_tracker='tensorboard',
                result_tracker_kwargs=dict(
                    experiment_path=experiment_dir + '/logs' + sub_dir_seed,
                ),
                training_kwargs=dict(num_epochs=config['training_kwargs']['num_epochs'],
                                     checkpoint_directory=experiment_dir + '/checkpoints' + sub_dir_seed,
                                     checkpoint_frequency=config['training_kwargs']['checkpoint_frequency'],
                                     checkpoint_name=config['training_kwargs']['checkpoint_name'],
                                     checkpoint_on_failure=config['training_kwargs']['checkpoint_on_failure'],
                                     batch_size=config['training_kwargs']['batch_size'],
                                     ),
                evaluator=config['evaluator'],
                evaluator_kwargs=dict(filtered=config['evaluator_kwargs']['filtered'],
                                      batch_size=config['evaluator_kwargs']['batch_size'], ),
                negative_sampler=config['negative_sampler'],
                negative_sampler_kwargs=dict(num_negs_per_pos=config['negative_sampler_kwargs']['num_negs_per_pos'], ),
                training_loop=config['training_loop'],
                stopper=config['stopper'],
                stopper_kwargs=dict(frequency=config['stopper_kwargs']['frequency'],
                                    patience=config['stopper_kwargs']['patience'],
                                    relative_delta=config['stopper_kwargs']['relative_delta'],
                                    evaluation_batch_size=config['evaluator_kwargs']['batch_size'], ),
                filter_validation_when_testing=True,
                use_testing_data=True,
                device=config['device'],
                random_seed=seed,
            )

    if test:
        evaluate_and_save_final_result(seeds, res, experiment_dir, config_path, sub_dir)


if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))  # kwargs
