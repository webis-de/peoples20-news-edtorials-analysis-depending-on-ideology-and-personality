import os, sys
import hashlib
from pathlib import Path


def save_json(df, name ):
    with open('data/'+name+'.json', 'w') as w:
        w.write(df.to_json(orient='records'))

def create_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)