import pandas as pd
import numpy as np
import math


def norm_ms2(ms2):
    """
    This function normalizes the intensities, as was done in SteroidXtract (Xing et. al)
    :param ms2: (np.array) 2d np.array
    :return: (np.array) The same array but with normalizes intesities
    """
    if ms2.size == 0:
        # print('No fragments in range')
        return ms2
    ms2[:, 1] = 100 * ms2[:, 1] / max(ms2[:, 1])
    ms2[:, 1] = np.sqrt(ms2[:, 1])

    return ms2


def ms2tobins(ms2, start_value, end_value, bin_width,
              df):
    """
    This function performs the bining of one spectrum and returning a row of df with the bins
    i is the row index of the data frame, thus the index of the spectrum. one row -->one spectrum

    :param ms2: (np.array) 2d array of m/z fragment and intesities
    :param start_value: (int) the minimal mass of a fragment, set to 50 Da for the given model
    :param end_value: (int) the maximal mass od a fragment, set to 550 and for the given model
    :param bin_width: (int) the mass fragments resolution, set to 0.1 Da for the given model
    :param df: (DataFrame) input DataFrame
    :return:
    """
    for j in range(ms2.shape[0]):
        bin_No = (end_value - start_value) / bin_width
        first_bin = int(df.shape[1] - bin_No)
        bin_position = math.floor((ms2[j, 0] - start_value) / bin_width) + first_bin
        df.iloc[0, bin_position] = max(ms2[j, 1], df.iloc[
            0, bin_position])  # if there is already an input of intensity in the bin, then only the maximal will be recorded
    df_updated = df.copy(deep=True)

    return df_updated


def get_ms2(mz_arr, in_arr, exact_mass, end_value, ms1_tol,
            start_value):
    """
     This function should cleans the spectrum and normalize the intensities.  for further processing it transforms the spectrum into 2 column array:
     1st column 'mz' 2nd 'Intensity'

    :param mz_arr: (np.array) array of mz fragments
    :param in_arr: (np.array) array of mz intensities
    :param exact_mass: (float) exact mass of parent ion, not mandatory
    :param end_value: (int) the maximal mass od a fragment, set to 550 and for the given model
    :param ms1_tol:(int) relevant if ExactMass exists. Fragments with mass higher than ms1_tol+ExactMass will be dropped
    :param start_value: (int) the minimal mass of a fragment, set to 50 Da for the given model
    :return: two numpy arrays of fragments and intensities
    """

    mz_arr = np.array(mz_arr, dtype=float)
    in_arr = np.array(in_arr, dtype=float)

    # Why do I use here 2d array?
    # In purpose to remove both fragment mass and intensity if one of them is not valid.
    # It is easier if I keep it in this format
    ms2 = np.array([mz_arr, in_arr])
    ms2 = ms2.transpose()

    if pd.isna(exact_mass) or exact_mass == 0: #if parent ion mass is missing
        pass
    else:
        exact_mass = float(exact_mass)
        ms2 = np.delete(ms2, obj=np.where(ms2[:, 0] > (exact_mass + ms1_tol)),
                        axis=0)  # Creating an array of mass fragments X2 ...

    # Each row is for a certain mass. col1 is the mass and col2 Intensity
    ms2 = np.delete(ms2, obj=np.where(ms2[:, 0] < start_value), axis=0)
    ms2 = np.delete(ms2, obj=np.where(ms2[:, 0] >= end_value), axis=0)
    ms2 = np.delete(ms2, obj=np.where(ms2[:, 1] == 1), axis=0)  # removing all the '1' intensities from mass bank data
    ms2_normed = norm_ms2(ms2)
    mz_arr = ms2_normed[:, 0]
    mz_arr = mz_arr.transpose()
    in_arr = ms2_normed[:, 1]
    in_arr = in_arr.transpose()

    return mz_arr, in_arr


def new_df(end_value, start_value, bin_width, input_df):  # creating an empty df with the all columns and bins
    """
    This function creates an empty df with bin columns

    :param end_value: (int) the maximal mass od a fragment, set to 550 and for the given model
    :param start_value: (int) the minimal mass of a fragment, set to 50 Da for the given model
    :param bin_width: (int) the mass fragments resolution, set to 0.1 Da for the given model
    :param input_df: (DataFrame) given input dataframe
    :return:
    """
    bin_No = (end_value - start_value) / bin_width
    mass_df = pd.DataFrame(columns=list(range(int(bin_No))))
    new_df = pd.concat([input_df, mass_df], axis=1)

    return new_df


def binn_spectrum(df, ms2, start_value, end_value, bin_width):
    """
    This function bins a single spectrum

    :param df: (DataFrame) input pandas dataframe
    :param ms2: (np.array) 2d array of m/z fragment and intesities
    :param start_value: (int) the minimal mass of a fragment, set to 50 Da for the given model
    :param end_value: (int) the maximal mass od a fragment, set to 550 and for the given model
    :param bin_width: (int) the mass fragments resolution, set to 0.1 Da for the given model
    :return: (DataFrame) returns the updated dataframe with the binned spectrum
    """
    binned_spectrum = ms2tobins(ms2, start_value, end_value, bin_width, df)
    binned_spectrum = binned_spectrum.fillna(0).infer_objects(copy=False)  # filling nans with zeros

    return binned_spectrum

def binning_df(df,bin_width, start_value, end_value,input_filename, ms1_tol):

    """
    this function performs the binning of all the spectra in the input DataFrame

    :param df: (DataFrame) input pandas dataframe
    :param bin_width: (int) the mass fragments resolution, set to 0.1 Da for the given model
    :param start_value: (int) the minimal mass of a fragment, set to 50 Da for the given model
    :param end_value: (int) the maximal mass od a fragment, set to 550 and for the given model
    :param input_filename: (str) name of the input
    :param ms1_tol: (int) relevant if ExactMass exists. Fragments with mass higher than ms1_tol+ExactMass will be dropped
    :return: (DataFrame) with binned spectra
    """

    df.reset_index(inplace = True)
    df_binned = pd.DataFrame()
    df = new_df(end_value, start_value, bin_width, df)  # this function adds the bin columns
    for row in range(df.shape[0]):
        df_row = df.loc[[row]]  # single row from the file DF
        df_row.reset_index(drop=True, inplace=True) # resetting index to avoid a mess, we don't need the index
        try:
            exact_mass = df_row.at[0,'ExactMass']
        except KeyError:
            print("rename the parent ion mass column to ExactMass if exists and rerun, the script can run without it, but then fragments with a mass higher than the parent ion will not be dropped")
            exact_mass = 0
        # here I will get the relevant fragments that passed the thresholds
        [mz_arr, in_arr] = get_ms2(df_row.at[0, 'mz'], df_row.at[0, 'Intensity'], exact_mass, end_value, ms1_tol, start_value)
        ms2 = np.array((mz_arr,in_arr))  # getting the spectrum as 2d array
        ms2 = ms2.transpose()  # transposing for convinience and the use of future functions
        if ms2.size == 0:
            print(f"No relevant fragments in file: {input_filename}  in DB: + {df_row.at[0, 'DB.']}  ")
            continue
        binned_spectrum = binn_spectrum(df_row, ms2, start_value, end_value,
                                        bin_width)  # binning the spectrum and attachning to the DF
        df_binned = pd.concat([df_binned, binned_spectrum], axis = 0,ignore_index=True)  # appending to previous rows
        print(f"row {row} was binned")

    return df_binned

