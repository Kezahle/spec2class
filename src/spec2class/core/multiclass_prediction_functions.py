import numpy as np
import pandas as pd
import pickle
from .utility_functions import save_df


def svm_pred(model_path, output_dir, df_name, df, chemclass_list):
    """

    :param model_path: (str) the path to the SVM trained model
    :param output_dir: (str) the path to the output
    :param df_name: (str) the name of the data set
    :param df: (DataFrame) with the prediction vectors
    :param chemclass_list: list of predicted chemical classes
    :return: (DataFrame) results df
    """

    df.reset_index(inplace = True, drop = True)
    X_test = np.asarray(df.select_dtypes(include=np.number))
    svm_model = pickle.load(open(model_path, 'rb'))
    pred_test_ix = svm_model.predict(X_test)  # Recieving predictions
    pred_prob_test = svm_model.predict_proba(X_test)
    top_3_pred_ix = np.argsort(pred_prob_test, axis=1)[:, -3:]
    top_3_pred_ix = top_3_pred_ix[:, ::-1]  # revert order
    results_df = pd.DataFrame \
        (columns= ['DB.' ,'final_pred','estimated_top2_pred' ,'estimated_top3_pred'
                 ,'probabilities'], dtype = 'object')
    for index in range(df.shape[0]):
        results_df.loc[index,'DB.'] = df.loc[index, 'DB.']
        results_df.loc[index,'final_pred'] = chemclass_list[pred_test_ix[index]]
        results_df.loc[index, 'estimated_top2_pred'] = chemclass_list[top_3_pred_ix[index][1]]
        results_df.loc[index, 'estimated_top3_pred'] = chemclass_list[top_3_pred_ix[index][2]]
        results_df.loc[index ,'probabilities'] = pred_prob_test[index][top_3_pred_ix[index]]
    save_df(df_name, output_dir, results_df)

    return results_df