import joblib
from sklearn.externals import joblib as old_joblib
import gzip
import pickle
import pandas as pd
import numpy as np
import os


def save_to_disk(dataset, filename, compress=3):
    """Save a dataset to file."""
    joblib.dump(dataset, filename, compress=compress)


def load_from_disk(filename):
    """Load a dataset from file."""
    name = filename
    if os.path.splitext(name)[1] == ".gz":
        name = os.path.splitext(name)[0]
    if os.path.splitext(name)[1] == ".pkl":
        return load_pickle_from_disk(filename)
    elif os.path.splitext(name)[1] == ".joblib":
        try:
            return joblib.load(filename)
        except KeyError:
            # Try older joblib version for legacy files.
            return old_joblib.load(filename)
        except ValueError:
            return old_joblib.load(filename)
    elif os.path.splitext(name)[1] == ".csv":
        # First line of user-specified CSV *must* be header.
        df = pd.read_csv(filename, header=0)
        df = df.replace(np.nan, str(""), regex=True)
        return df
    else:
        raise ValueError("Unrecognized filetype for %s" % filename)


def load_pickle_from_disk(filename):
    """Load dataset from pickle file."""
    if ".gz" in filename:
        with gzip.open(filename, "rb") as f:
            df = pickle.load(f)
    else:
        with open(filename, "rb") as f:
            df = pickle.load(f)
    return df