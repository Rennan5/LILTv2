import argparse
import datetime
import json
import os
import shutil
import random
import subprocess

def run_trial(_hyperparams, _arguments, _save_path):
    """ Run single trial """
    commands = [
        "python3", "train.py",
        "--model-type", _arguments.model_type,
        "--epochs", str(_hyperparams['epochs']),
        "--learning-rate", str(_hyperparams['learning_rate']),
        "--batch-size", str(_hyperparams['batch_size']),
        "--patience", str(_hyperparams['patience']),
        "--dataset", str(_arguments.dataset),
        "--output-path", _save_path
    ]

    subprocess.run(
        commands, check=True,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL
    )

def get_random_lr():
    """ Get random learning rate """
    return 10 ** (-1 * random.uniform(4, 6))


def get_hyperparams_possibilities(number_of_possibilities):
    """ Get n hyperparams possibilities """
    hyperparams_possibilities = []

    for _ in range(number_of_possibilities):
        '''
        hyperparams_possibility = {
            "epochs": random.choice([100, 200]),
            # "epochs": random.choice([2]),
            "learning_rate": get_random_lr(),
            "batch_size": random.choice([2, 4]),
            "patience": 5,
        }
        '''
        hyperparams_possibility = {
            "epochs": random.choice([100, 200]),
            # "epochs": random.choice([2]),
            "learning_rate": 1e-4,
            "batch_size": random.choice([2, 4]),
            "patience": 5,
        }
        hyperparams_possibilities.append(hyperparams_possibility)

    return hyperparams_possibilities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--number-of-trials', type=int, default=10)
    arguments = parser.parse_args()

    # Setup variables
    output_path = os.path.join(
        'output', f'{arguments.dataset}_LILT_model_' +
            datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '_optimized'
    )

    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, 'params.json')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({
            'dataset': arguments.dataset,
            'model-type': arguments.model_type,
            'number-of-trials': arguments.number_of_trials,
        }, indent=4))

    for i, hyperparams in enumerate(get_hyperparams_possibilities(
            arguments.number_of_trials)):
        print(f'trial { i }\n{ hyperparams }')

        save_path = os.path.join(output_path, f'trial_{i}')

        run_trial(hyperparams, arguments, save_path)
        shutil.rmtree(os.path.join(save_path, "checkpoints"))
