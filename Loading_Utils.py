import torch
from pykeen.datasets import FB15k237, WN18RR
from pykeen.models import BoxE, RotatE
from pykeen.triples import TriplesFactory

import json
from ExpressivEModel import ExpressivE


def load_checkpoint(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    checkpoint = torch.load(checkpoint_path)

    if config['dataset'] == 'FB15k237':
        dataset = FB15k237()
    elif config['dataset'] == 'WN18RR':
        dataset = WN18RR()
    else:
        raise Exception('Dataset %s unknown!' % config['dataset'])

    triples_factory = TriplesFactory(
        mapped_triples=dataset.training.mapped_triples,
        relation_to_id=checkpoint['relation_to_id_dict'],
        entity_to_id=checkpoint['entity_to_id_dict'],
        create_inverse_triples=config['dataset_kwargs']['create_inverse_triples'],
    )

    if config['loss'] == 'NSALoss' or config['loss'] == 'NSSALoss':
        loss_str = 'nssa'
    else:
        raise Exception('Unknown loss \'%s\'' % config['loss'])

    if config['model'] == 'ExpressivEModel':

        if 'interactionMode' in config['model_kwargs']:
            trained_model = ExpressivE(triples_factory=triples_factory,
                                       embedding_dim=config['model_kwargs']['embedding_dim'],
                                       p=config['model_kwargs']['p'],
                                       min_denom=config['model_kwargs']['min_denom'],
                                       tanh_map=config['model_kwargs']['tanh_map'],
                                       interactionMode=config['model_kwargs']['interactionMode'],
                                       loss=loss_str,
                                       loss_kwargs=dict(
                                           reduction=config['loss_kwargs']['reduction'],
                                           margin=config['loss_kwargs']['margin'],
                                           adversarial_temperature=config['loss_kwargs']['adversarial_temperature'])
                                       )
        else:
            trained_model = ExpressivE(triples_factory=triples_factory,
                                       embedding_dim=config['model_kwargs']['embedding_dim'],
                                       p=config['model_kwargs']['p'],
                                       min_denom=config['model_kwargs']['min_denom'],
                                       tanh_map=config['model_kwargs']['tanh_map'],
                                       loss=loss_str,
                                       loss_kwargs=dict(
                                           reduction=config['loss_kwargs']['reduction'],
                                           margin=config['loss_kwargs']['margin'],
                                           adversarial_temperature=config['loss_kwargs']['adversarial_temperature'])
                                       )
    elif config['model'] == 'BoxE':
        trained_model = BoxE(triples_factory=triples_factory,
                             embedding_dim=config['model_kwargs']['embedding_dim'],
                             p=config['model_kwargs']['p'],
                             loss=loss_str,
                             loss_kwargs=dict(
                                 reduction=config['loss_kwargs']['reduction'],
                                 margin=config['loss_kwargs']['margin'],
                                 adversarial_temperature=config['loss_kwargs'][
                                     'adversarial_temperature']),
                             )
    elif config['model'] == 'RotatE':
        trained_model = RotatE(triples_factory=triples_factory,
                               embedding_dim=config['model_kwargs']['embedding_dim'],
                               loss=loss_str,
                               )

    trained_model.load_state_dict(checkpoint['model_state_dict'])

    return config, dataset, trained_model
