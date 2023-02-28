import os
import sys

from Analysis_Utils import get_evaluator_from_config, parse_directory
from Loading_Utils import load_checkpoint
import json


def evaluation_per_relation(model_dir, config_path, sub_dir=None):
    checkpoint_path, analysis_path = parse_directory(model_dir, sub_dir)
    config, dataset, trained_model = load_checkpoint(config_path, checkpoint_path)
    evaluator = get_evaluator_from_config(config)

    result_per_relation = {}

    # Perform an evaluation for each relation:
    for relation_name in dataset.relation_to_id.keys():
        # Filter training, testing and validation set by the name of the relation
        training_for_curr_rel = dataset.training.new_with_restriction(
            relations={relation_name},
        )
        validation_for_curr_rel = dataset.validation.new_with_restriction(
            relations={relation_name},
        )
        testing_for_curr_rel = dataset.testing.new_with_restriction(
            relations={relation_name},
        )

        # Perform the evaluation per relation
        test_result_of_curr_relation = evaluator.evaluate(
            model=trained_model,
            mapped_triples=testing_for_curr_rel.mapped_triples,
            additional_filter_triples=[
                training_for_curr_rel.mapped_triples,
                validation_for_curr_rel.mapped_triples,
            ],
            batch_size=config['evaluator_kwargs']['batch_size'],
        )

        result_per_relation[relation_name] = test_result_of_curr_relation.to_flat_dict()

    return result_per_relation


def main(**kwargs):
    experiment_dir = './Benchmarking'

    if 'model_dir' in kwargs.keys():
        model_dir = kwargs['model_dir']
    else:
        raise Exception('The <model_dir> parameter needs to be specified.')

    if 'sub_dir' in kwargs.keys():
        sub_dir = kwargs['sub_dir']
    else:
        raise Exception('The <sub_dir> parameter needs to be specified.')

    if 'config_path' in kwargs.keys():
        config_path = kwargs['config_path']
    else:
        raise Exception('The <config_path> parameter needs to be specified.')

    if 'gpu' in kwargs.keys():
        gpu = kwargs['gpu']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    res = evaluation_per_relation(model_dir, config_path, sub_dir)

    saving_directory = os.path.join(experiment_dir, 'per_relation_result', sub_dir)

    if not (os.path.exists(saving_directory)):
        os.makedirs(saving_directory)

    with open(os.path.join(saving_directory, 'result.json'), 'w') as outfile:
        json.dump(res, outfile)


if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))  # kwargs
