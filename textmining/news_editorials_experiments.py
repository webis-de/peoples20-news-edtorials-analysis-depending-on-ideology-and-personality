import itertools
import textmining.machine_learning as machine_learning
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from functools import reduce #python 3
import time
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score, f1_score
import operator
import pickle
from pathlib import Path


def get_features_map(df):
    cols_map = {
        'liwc': [l for l in df.columns if l.startswith('liwc_')],
        'nrc': [l for l in df.columns if l.startswith('nrc_')],
        'mpqa_arg': [l for l in df.columns if l.startswith('mpqa_arg_')],
        'mpqa_subjobg': [l for l in df.columns if l.startswith('mpqa_subjobg_')],
        'adu': [l for l in df.columns if l.startswith('adu_')],
        'lemma': [l for l in df.columns if l.startswith('lemma')],
    }
    if 'lemma' in cols_map['lemma']:
        cols_map['lemma'].remove('lemma')
    return cols_map

def get_all_feature_types_comb(df):
    cols_map = get_features_map(df)
    combs=[]
    for i in range(1, len(cols_map.keys())+1):
        combs.extend(list(itertools.combinations(cols_map.keys(), i)))
    return combs




def get_x_y(df, y, remove_outliers = True,  normalizing_method = "standard"):
    df = df.fillna(0)
    df_train = df[df['split_label'] == 'train']
    df_test = df[df['split_label'] == 'test']
    ## Remove outliers:
    cols = get_features_map(df_train).values()
    cols = reduce(lambda x,y: x+y,cols)
    if remove_outliers:
        print("removing outliers by clipping values...")
        
        #df_train_features = df_train[cols]
        df_train, df_test = machine_learning.clip_outliers(df_train, df_test, lower_percentile=1,  upper_percentile=99)
    
    ## X Y
    print("getting X y data...")
    X_train = df_train[cols]    
    y_train = df_train[y].values
    
    X_test = df_test[cols]
    y_test = df_test[y].values  
    
    ## Scale - normalize
    X_train, X_test = machine_learning.normalize(X_train, X_test, normalizing_method=normalizing_method)
        
    print('end of get_x_y.')
    return X_train, y_train, X_test, y_test

def get_instances_with_features_sub(X_train_df, X_test_df, features):
    cols_map = get_features_map(X_train_df)
    cols= []
    for f in features:
        cols.extend(cols_map[f])
    #print(cols)
    X_train = X_train_df[cols].values
    X_test = X_test_df[cols].values    
    
    return X_train, X_test





def validate_train_test(X_train_df, y_train, X_test_df, y_test, feature_types, validation_folds=5):
    X_train, X_test = get_instances_with_features_sub(X_train_df, X_test_df, feature_types)
    #if 'liwc' in feature_types and 'nrc' in feature_types and 'mpqa_arg' in feature_types and 'mpqa_subjobg' in feature_types and 'adu' in feature_types:
     #   best_params = {}
    #else:
    best_params = machine_learning.svc_param_gridsearch(X_train, y_train, nfolds_or_division=validation_folds)

    result = machine_learning.train_test(X_train, y_train, X_test, y_test, params=best_params)
    result['params'] = best_params
    return result


def run_experiments(df, ideologies, filename=None,
                    remove_outliers=True,
                    normalize="sqrt"):
    #ideologies=['liberal_majority', 'conservative_majority']
    r = {}
    print('running experiments for ideologies: ', ideologies,
          '\n remove_outliers: ',  remove_outliers,
          '\n normalize: ', normalize
          )
    for ideology in ideologies:
        ## FOR EACH IDEOLOGY
        # For all combination
        print("preprocessing data...")
        X_train_df, y_train, X_test_df, y_test = get_x_y(df, ideology, remove_outliers=remove_outliers,  normalizing_method=normalize)
        print("END of preprocessing")

        results = []
        print(ideology)
        print('+++++++++++++++++++++++++++++++++++++++++++++')
        all_feature_types_comb = get_all_feature_types_comb(df)

        for feature_types in all_feature_types_comb:
            result = {}
            print(feature_types)
            #if len(feature_types) == 4 and  "'liwc', 'nrc', 'mpqa_subjobg', 'lemma')" in str(feature_types):
            #    continue
                #('liwc', 'nrc', 'mpqa_subjobg', 'lemma')
            start_time = time.time()
            result = validate_train_test(X_train_df, y_train, X_test_df, y_test, feature_types, validation_folds=5)

            result['features'] = str(feature_types)
            result['ideology'] = ideology
            results.append(result)
            elapsed_time = time.time() - start_time
            print( 'macro-f1: ', result['macro'], 'time(s): ', round(elapsed_time, 3))
            print('-------------------------------------------')

        r[ideology] = pd.DataFrame.from_dict(results)

        if filename is not None:
            r[ideology].sort_values(by=['macro'], ascending=False).to_csv('{}_{}.csv'.format(filename, ideology))
    return r

def train_baseline(df, ideology , strategy= 'uniform'): #'conservative_majority'
    X_train_df, y_train, X_test_df, y_test = get_x_y(df, ideology)
    return machine_learning.dummy_train_test(X_train_df, y_train, X_test_df, y_test, strategy=strategy)

############### USED FOR CHECKING SIGNIFICANCY BETWEEN THE MODELS

def save_model(df, feature_types, ideology, remove_outliers = True,  normalizing_method="sqrt", nfolds_or_division=5):
    str_features = normalizing_method+'_'+'-'.join([str(x) for x in feature_types])
    Path('../out/models/'+ideology+'/').mkdir(parents=True, exist_ok=True)
    pkl_filename = '../out/models/'+ideology+'/'+str_features+'.pkl'

    X_train_df, y_train, X_test_df, y_test = get_x_y(df, ideology,   remove_outliers = remove_outliers,  normalizing_method=normalizing_method)
    X_train, X_test = get_instances_with_features_sub(X_train_df, X_test_df,  feature_types)
    best_params = machine_learning.svc_param_gridsearch(X_train, y_train, nfolds_or_division=nfolds_or_division)
    print('saving file: ', pkl_filename)
    machine_learning.train_save(X_train, y_train, pkl_filename, params=best_params)




def dependent_pairs(dv1, dv2, alpha=0.05):
    # Check if they are normally distriuted
    diff = list(map(operator.sub, dv1, dv2))
    is_normal = stats.shapiro(diff)[1] > 0.05
    #is_normal = False
    stat, p_val = stats.ttest_rel(dv1, dv2) #if is_normal else stats.wilcoxon(dv1, dv2)
    wilk_stat, wilk_p_val =  stats.wilcoxon(dv1, dv2)
    
    return stat, p_val, is_normal, wilk_stat, wilk_p_val


def run_experiments_with_test_repetition(df, all_feature_types_comb, ideology,  n_splits=5, score='macro', normalizing_method="sqrt"):
    result = {}
    r = []
    ## FOR EACH IDEOLOGY
    # For all combination#

    print("preprocessing data...")
    X_train_df, y_train, X_test_df, y_test = get_x_y(df, ideology,  remove_outliers=True, normalizing_method=normalizing_method)
    print("END of preprocessing")

    print(ideology)
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    # all_feature_types_comb = get_all_feature_types_comb(df)
    print(all_feature_types_comb, len(all_feature_types_comb))
    for feature_types in all_feature_types_comb:

        print(feature_types)

        start_time = time.time()
        if 'dummy' in feature_types[0] :
            f = all_feature_types_comb[1]
            X_train, X_test = get_instances_with_features_sub(X_train_df, X_test_df, f)

        else:
            X_train, X_test = get_instances_with_features_sub(X_train_df, X_test_df, feature_types)

        ## 1. svm search grid - Leave one out validation
        ## get best param the test on test set 10 folds
        # print(best_params)
        elapsed_time = time.time() - start_time

        skf = StratifiedKFold(n_splits=n_splits)
        skf.get_n_splits(X_test, y_test)

        runs = []
        for _, test_index in skf.split(X_test, y_test):
            X_sub_test = X_test[test_index]

            y_sub_test = y_test[test_index]

            if len(X_sub_test) != 0:
                if 'dummy' in feature_types[0]:
                    macro = machine_learning.dummy_train_test(X_train, y_train, X_sub_test, y_sub_test,
                                                              strategy='uniform')[score]
                else:
                    str_features = normalizing_method +'_' +'-'.join([str(x) for x in feature_types])
                    pkl_filename = '../out/models/' + ideology + '/' + str_features + '.pkl'

                    with open(pkl_filename, 'rb') as file:
                        # print('loading model: ' + pkl_filename)
                        pickle_model = pickle.load(file)
                        y_pred = pickle_model.predict(X_sub_test)
                        macro = f1_score(y_pred=y_pred, y_true=y_sub_test, average=score)

                runs.append(macro)
                elapsed_time = time.time() - start_time

        result[str(feature_types)] = runs
        r.append(runs)
        # print(n, ' - ', feature_types, ' ', results)

    stat, p_val, is_normal, wilk_stat, wilk_p_val = dependent_pairs(r[0], r[1])
    # r[ideology] = pd.DataFrame.from_dict(results)
    # if filename is not None:
    #    r[ideology].sort_values(by=['macro'], ascending=False).to_csv('{}_{}.csv'.format(filename, ideology))
    return stat, p_val, is_normal, wilk_stat, wilk_p_val, result