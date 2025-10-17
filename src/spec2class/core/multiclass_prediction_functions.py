import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def svm_pred(model_path, output_dir, df_name, df, chemclass_list, output_format="csv"):
    """
    Perform SVM prediction and save results in specified format

    :param model_path: (str) the path to the SVM trained model
    :param output_dir: (str) the path to the output
    :param df_name: (str) the name of the data set
    :param df: (DataFrame) with the prediction vectors
    :param chemclass_list: list of predicted chemical classes
    :param output_format: (str) output format - 'csv', 'tsv', 'pickle', or 'all'
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
    
    # Save in requested format(s)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if output_format == "all":
        # Save in all formats
        results_df.to_csv(output_path / f"{df_name}.csv", index=False)
        results_df.to_csv(output_path / f"{df_name}.tsv", sep='\t', index=False)
        results_df.to_pickle(output_path / f"{df_name}.pkl")
        print(f"Saved {df_name} in all formats")
    elif output_format == "csv":
        results_df.to_csv(output_path / f"{df_name}.csv", index=False)
        print(f"Saved {df_name}.csv")
    elif output_format == "tsv":
        results_df.to_csv(output_path / f"{df_name}.tsv", sep='\t', index=False)
        print(f"Saved {df_name}.tsv")
    elif output_format == "pickle":
        results_df.to_pickle(output_path / f"{df_name}.pkl")
        print(f"Saved {df_name}.pkl")

    return results_df