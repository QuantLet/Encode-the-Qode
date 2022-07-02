import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess
import shutil
import pathlib
import argparse


def parse_folder(path: str): 
    path = os.walk(path)    

    for root, directories, files in path:
        for directory in directories:
            print(directory)
        for file in files:
            print(file)


def main():
    pass

if __name__ == "__main__":
    main()