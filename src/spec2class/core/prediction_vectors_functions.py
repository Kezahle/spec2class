import torch
import numpy as np
import pandas as pd
import os
import sys
from .dataset_class import Dataset
from .utility_functions import get_net


def test_model_1_batch(trained_model_path, df_binned, net, bin_no, device):

    """
    get neural net's prediction in a single batch, w/o the use of dataloader

    :param trained_models_path: (str)
    :param df_binned: (pd.DataFrame) test set
    :param net: (object) CNN object
    :param bin_no: (int) length of the binned spectrum
    :param device: (str) cuda or cpu
    :return: y_pred (np.array) -  model's predictions
    """

    X = np.asarray(df_binned[range(bin_no)], dtype = float)
    X = torch.from_numpy(np.reshape(X, (X.shape[0], 1, bin_no))).to(device)
    X = X.to(dtype = torch.float64)
    net.eval()
    net.load_state_dict(torch.load(trained_model_path, map_location= device))
    y_pred = net(X)
    y_pred = y_pred.cpu()
    y_pred = y_pred.detach().numpy()
    y_pred = y_pred.ravel()  # to a row vector

    return y_pred


def create_dataloader(X_data, DBs_list ,params):
    """
    This function creates a generator with given indices that loads the data in batches
    :param data_df: (pd.DataFrame) training set
    :param X_data: (np.array) a matrix of the all training set binned spectra
    :param params: (dict) parameters for DataLoader
    :return: generator

    """
    ix = range(X_data.shape[0])
    x_dict = {index: X_data[index].transpose() for indices, index in enumerate(ix)}  #Keys are indices and binned vectors are values
    set = Dataset(DBs_list, x_dict) # I use the subset only for the purpose of using the list of indices for train/test sets.
    dataloader = torch.utils.data.DataLoader(set, **params)

    return dataloader


def test_model_batches(trained_model_path,net, device, dataloader, bin_no = 5000):
    """
    This function gets the binary nn model prediction using a dataloader. The dataloader is
    an instanse of the DataLoader pytoch class. Loading the data in batches accelarate the process
    dramatically when GPU is used.

    :param trained_model_path: (str)
    :param net: (object) CNN object
    :param device: (str) cuda or cpu
    :param dataloader: input data generator object
    :param bin_no: (int) length of the binned spectrum
    :return: y_pred: (float) a nunpy array of the model's prediction, DBs : (str) a list of the corresponding sample DB
    entries
    """

    y_pred = []
    DBs = []
    net.eval()
    net.load_state_dict(torch.load(trained_model_path, map_location=device))
    with torch.no_grad():
        for x_batch, DB_batch in dataloader:
            X = np.reshape(x_batch, (x_batch.shape[0], 1, bin_no)).to(device)
            X = X.to(dtype=torch.float64)
            y_pred_batch = net(X)
            y_pred_batch = y_pred_batch.cpu()
            y_pred_batch = y_pred_batch.detach().numpy()
            y_pred_batch = y_pred_batch.ravel()  # to a row vector
            y_pred = np.concatenate((y_pred, y_pred_batch), axis = 0)
            DBs = np.concatenate((DBs, DB_batch), axis = 0)

    return y_pred, DBs



def get_pred_vec_dfs(chemclass_list, bin_no, df, net, models_dir, device, params):
    """
    This function produces two data frames with chemical class names as columns and rows according to the spectra.

    :param chemclass_list: (list) list of the predicted classes
    :param bin_no: (int) length of the binned spectrum vector
    :param df: (pd.DataFrame) input spectra df
    :return:  (pd.DataFrame) y_pred_df -prediction vectors
    """
    y_pred_df = pd.DataFrame()
    y_pred_df['DB.'] = df['DB.']
    X_data = np.asarray(df[range(bin_no)], dtype=float)
    dataloader = create_dataloader(X_data, y_pred_df['DB.'].to_list(), params)
    for chemclass in chemclass_list:
        print(chemclass)
        trained_model_file = f'trained_model_{chemclass}_no_x_val.pt'
        trained_model_path = os.path.join(models_dir,trained_model_file)
        y_pred, DBs = test_model_batches(trained_model_path,net, device, dataloader, bin_no)
        if DBs.tolist() != y_pred_df['DB.'].to_list():
            print('The DBs are not matching, check if the dataloader shuffles the batches')
            return
        y_pred_df[chemclass] = y_pred
        print(f"{chemclass} done")

    return y_pred_df


def get_pred_vecs(chemclass_list, df_name, df, output_path, end_value, start_value, bin_width, net_path, net_dir,
         dropout_conv, dropout_linear, models_dir, params):
    """
    This is a wrapper function to get_pred_vec_dfs function that generated all the predictions from the binary models
    for all the spectra in the input set

    :param chemclass_list: (list) list of all predicted classes
    :param df_name: (str) name of the input set
    :param df: (DataFrame) input DataFrame
    :param output_path: (str) where the prediction vectors will be saved
    :param end_value:  (int) the maximal mass od a fragment, set to 550 and for the given model
    :param start_value: (int) the minimal mass of a fragment, set to 50 Da for the given model
    :param bin_width: (int) the mass fragments resolution, set to 0.1 Da for the given model
    :param net_path: (str) the path to the neural net class
    :param net_dir: (str) the directory where the neural net class is saved
    :param dropout_conv: (int) dropout ratio (used to initialize the NN but not used in inference)
    :param dropout_linear: (int) dropout ratio (used to initialize the NN but not used in inference)
    :param models_dir: (str) directory of the trained binary models
    :param params: (dict) parameters for inference in batches
    :return: (DataFrame) with the predictions from all binary models
    """

    bin_no = int((end_value - start_value) / bin_width)
    df.reset_index(inplace=True, drop=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda available')
    else:
        device = torch.device('cpu')
    sys.path.append(net_dir)
    net, _ = get_net(end_value, start_value, bin_width, net_path, dropout_conv,
                                dropout_linear, device)  # init NN

    y_pred_df = get_pred_vec_dfs(chemclass_list, bin_no,
                                            df, net, models_dir, device, params)  # getting predictions for all chemical classes
    
    # Removed automatic save_df call - saving is now handled by svm_pred based on output_format
    print('Prediction vectors done')
    return y_pred_df