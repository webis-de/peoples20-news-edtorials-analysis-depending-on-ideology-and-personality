
import itertools
import pandas as pd
from ast import literal_eval
from pathlib import Path
import textmining.news_editorials_experiments as experiment

def _features_type(row):
    features = row['features'].replace("'", "").replace(' ', '').strip(')(').split(',')
    features = [x for x in features if len(x.strip())>0]
    type_ = 'style'
    if 'lemma' in features:
        type_ = 'content'  if (len(features) == 1 ) else 'content+style'
    row['single'] = len(features) == 1
    row['type'] = type_
    return row

def get_top_features_per_type(results):
    best = {}
    for type_, type_df in results.groupby(['type']):
        best[type_] =type_df[type_df['macro']== type_df['macro'].max()]  if type_ != 'content' else type_df
    return best

def get_list_from_df_cell(df_col):
    return [literal_eval(x) for x in df_col.values.tolist()]


def get_combinations(df_dic):
    all_features = [('lemma',)]

    style = get_list_from_df_cell(df_dic['style']['features'])
    all_features.extend(style)

    style_content = get_list_from_df_cell(df_dic['content+style']['features'])
    all_features.extend(style_content)

    comb = list(itertools.combinations(([('dummy',)]+all_features), 2))

    return all_features, comb


def compare_models(df, var, models_comb, normalizing_method, name=None):
    results = []
    for pair in models_comb:
        stat, p_val, is_normal, wilk_stat, wilk_p_val, data = experiment.run_experiments_with_test_repetition(df,  pair,  var,
                                                                                       normalizing_method=normalizing_method)
        result = {}
        result['model_pair'] = pair
        result['is_normal'] = is_normal
        result['stat'] = stat
        result['p_val'] = p_val
        result['wilk_stat'] = wilk_stat
        result['wilk_p_val'] = wilk_p_val
        result['significant'] = p_val <0.05 if is_normal else wilk_p_val<0.05
        result['data'] = data
        results.append(result)
    result_df =pd.DataFrame(results)
    Path('../out/model_pair_comparison/').mkdir(parents=True, exist_ok=True)

    if name is not None: result_df.to_csv('../out/model_pair_comparison/'+name)

    return result_df

from os import path

def get_top_features_from_path(result_path):
    models_results = pd.read_csv(result_path)
    models_results = models_results.apply(_features_type, axis=1)  # add type : content style

    features_per_best_model = get_top_features_per_type(models_results)
    return models_results, features_per_best_model

def run_model_pairs_significance(result_path, data,ideology, normalization, name ):
    models_results = pd.read_csv(result_path)
    models_results = models_results.apply(_features_type, axis=1) # add type : content style

    features_per_best_model = get_top_features_per_type(models_results)
    features, model_pair_comb = get_combinations(features_per_best_model)

    for feature_set in features:
        str_features = normalization + '_' + '-'.join([str(x) for x in feature_set])
        pkl_filename = '../out/models/' + ideology + '/' + str_features + '.pkl'
        if not path.exists(pkl_filename):
            print('model pickle was not created. Creating it now')
            experiment.save_model(data, feature_set, ideology, normalizing_method=normalization)

    comparison_df = compare_models(data, ideology, model_pair_comb, normalization, name=name)
    return comparison_df

