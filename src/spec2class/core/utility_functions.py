import pandas as pd
import numpy as np
import torch
import sys
import os
import torch.nn as nn
import importlib
from pandas.api.types import is_string_dtype




def import_nn(net_filename, net_path):
    """
    importing nerual net from path
    :param net_filename: (str)
    :param net_path: (str)
    :return: (object) neural net
    """
    spec = importlib.util.spec_from_file_location(net_filename, net_path)
    nn_model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = nn_model
    spec.loader.exec_module(nn_model)

    return nn_model


def get_net(end_value, start_value, bin_width, net_path, dropout_conv, dropout_linear, device):

    """
    The function initializes the CNN with the given parameters

    :param end_value (int): maximal fragment's mass
    :param start_value (int): minimal fragment's mass
    :param bin_width (int): mass difference between the bins of the binned spectrum
    :param net_path (str):the path to the CNN class
    :param dropout_conv (float): #dropout portion post convolution layers in the model
    :param dropout_linear (float): #dropout portion post linear layers in the model
    :return: net (object): CNN initialized class
             net_filename: the name of the model according to filename
    """
    bin_no = int((end_value - start_value) / bin_width)
    net_filename = os.path.split(net_path)[1]  # dropping '.py'
    net_filename = net_filename.replace('.py', '')
    nn_model = import_nn(net_filename, net_path)  # importing nn from path
    net = nn_model.Net(bin_no, dropout_conv, dropout_linear)
    net.to(dtype=torch.float64)
    net.to(device)

    return net, net_filename


def save_df(filename, file_dir, df):
    """
    This is a function that saves a data frame in all usual formats
    :param filename: (str)
    :param file_dir: (str)
    :param df: (DataFrame)
    :return: None
    """
    save_dir = os.path.join(file_dir,filename)
    df.to_pickle(save_dir + '.pkl', compression = None)
    df.to_csv(save_dir+'.csv')
    df.to_csv(save_dir+'.tsv', sep = '\t')
    print(f"Saved {filename} ")

def make_folder(folder_path, folder_name):
    """

    :param folder_path: (str) a path to the new folder
    :param folder_name: (str) the name of the new folder
    :return: (str) the new created directory
    """
    new_folder_dir = os.path.join(folder_path, folder_name)
    if os.path.isdir(new_folder_dir):
        pass
    else:
        os.makedirs(new_folder_dir)

    return new_folder_dir

def read_df_and_format_mz_intensity_arrays_old(df_path):
    """
    This function reads df in .csv/.tsv or .pkl formats and converts the mz and Intensity lists to np arrays
    :param df_path: path to the input df
    :return:
    """

    if df_path.endswith(".csv"):
        df = pd.read_csv(df_path)

    elif df_path.endswith(".tsv"):
        df = pd.read_csv(df_path, sep='\t')

    elif df_path.endswith(".pkl"):
        df = pd.read_pickle(df_path)
        # reforamting strings to arrays
        if is_string_dtype(df.loc[df.index[0], 'Intensity']):
            df['Intensity'] = df['Intensity'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=float))

        if is_string_dtype(df.loc[df.index[0], 'mz']):
            df['mz'] = df['mz'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=float))

    else:
        print(f"use different format for {df_path}")

    # reforamting strings to arrays
    if df_path.endswith(".csv") or df_path.endswith(".tsv"):
        if is_string_dtype(df['Intensity']):
            df['Intensity'] = df['Intensity'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype = float))

        if is_string_dtype(df['mz']):
            df['mz'] = df['mz'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ',  dtype = float))

    return df
    


def _to_float_array(x):
    """
    Convert one cell value to a 1-D NumPy float array.
    Accepts strings like "[1.0 2.0]", Python lists,
    tuples, ndarrays, scalars, or NaN.
    """
    if isinstance(x, str):
        # "[100.1 101.2]"  ?  array([100.1, 101.2])
        cleaned = x.strip("[]").strip()
        if not cleaned:
            return np.array([], dtype=float)
        try:
            return np.array([float(val) for val in cleaned.split()], dtype=float)
        except (ValueError, AttributeError):
            return np.array([], dtype=float)

    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)

    # scalar (int/float) or NaN ? wrap in array or empty array
    if pd.isna(x):
        return np.array([], dtype=float)      # keep missing spectra empty
    return np.asarray([x], dtype=float)       # single-number spectrum




def read_df_and_format_mz_intensity_arrays(path):
    """
    Read a spectra table (.csv, .tsv, .pkl) and ensure the
    `mz` and `Intensity` columns contain NumPy float arrays.
    """
    _, ext = os.path.splitext(path.lower())

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext == ".pkl":
        df = pd.read_pickle(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'; expected .csv, .tsv, or .pkl"
        )

    for col in ("Intensity", "mz"):
        if col in df.columns:
            df[col] = df[col].apply(_to_float_array)

    return df 
        
      