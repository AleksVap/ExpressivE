import json
import os

from pykeen.evaluation import RankBasedEvaluator
from Loading_Utils import load_checkpoint


def parse_directory(model_dir, sub_dir=None):
    if sub_dir is None:
        checkpoint_path = '%s/checkpoints/checkpoint.pt' % model_dir
        analysis_path = '%s/analysis' % model_dir
    else:
        checkpoint_path = '%s/checkpoints/%s/checkpoint.pt' % (model_dir, sub_dir)
        analysis_path = '%s/analysis/%s/' % (model_dir, sub_dir)

    return checkpoint_path, analysis_path


def get_evaluator_from_config(config):
    if config['evaluator'] == 'RankBasedEvaluator':
        evaluator = RankBasedEvaluator(
            filtered=config['evaluator_kwargs']['filtered'],
        )
    else:
        raise Exception('Unknown evaluator \'%s\'!' % config['evaluator'])

    return evaluator


def analyze_checkpoints(model_dir, config_path, sub_dir=None):
    checkpoint_path, analysis_path = parse_directory(model_dir, sub_dir)
    config, dataset, trained_model = load_checkpoint(config_path, checkpoint_path)
    evaluator = get_evaluator_from_config(config)

    final_val_results = evaluator.evaluate(
        model=trained_model,
        mapped_triples=dataset.validation.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
        ],
        batch_size=config['evaluator_kwargs']['batch_size'],
    )

    final_test_results = evaluator.evaluate(
        model=trained_model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
        batch_size=config['evaluator_kwargs']['batch_size'],
    )

    # Save Results:
    print('[Info] Saving results!')
    final_val_results = final_val_results.to_flat_dict()
    final_test_results = final_test_results.to_flat_dict()

    val_path = os.path.join(analysis_path, 'val_res.json')
    test_path = os.path.join(analysis_path, 'test_res.json')

    res = dict(res=[final_val_results, final_test_results], dir_path=[analysis_path, analysis_path],
               file_path=[val_path, test_path])

    for i in range(len(res['res'])):
        path = res['dir_path'][i]
        file_path = res['file_path'][i]
        r = res['res'][i]
        if not os.path.exists(path):
            os.makedirs(path)
        with open(file_path, 'w') as fp:
            json.dump(r, fp)

    return {'val_res': final_val_results, 'test_res': final_test_results}
