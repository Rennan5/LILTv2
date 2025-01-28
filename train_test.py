import argparse
import datetime
import os

import numpy as np

from utils.dataset import load_cord_dataset, load_custom_dataset, load_mixed_dataset
from utils.liltv2.model import LiltModel
from utils.save import save_metrics, save_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    dataset = f'datasets/{dataset}'
    
    print(dataset)
