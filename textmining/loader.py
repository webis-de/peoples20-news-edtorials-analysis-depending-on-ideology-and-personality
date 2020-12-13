
import pandas as pd
import numpy as np
import platform
from functools import reduce
import textmining.machine_learning as machine_learning
import os
class loader():
    lean_ideologies = ['Market Skeptic Republicans', 'New Era Enterprisers', 'Disaffected Democrats']
    extreme_ideologies = ['Country First Conservatives', 'Opportunity Democrats', 'Solid Liberals']
    effect_mapping = {1:'strongly_challenging', 2:'challenging', 3:'no_effect', 4:'reinforcing', 5:'empowering'}
    abstract_effect_mapping = {1:'challenging', 2:'no_effect', 3:'reinforcing'}

    def __init__(self):
        print('"corpus" is set. It contains the 6000 annotation')
        self.corpus = self.load_corpus()
        self.liberal = None
        self.conservative = None
        self.data_division = None
        self.pars_features_df = None
        self.discourse_adus_df = None
        self.data_division_df = None
        self.df_features = None

    @staticmethod
    def get_data_root():
        root = 'ROOTPATH'
        return root


    def load_corpus(self):
        # Download the json fole for webis-editorial-quality-18
        path_corpus = loader.get_data_root() + 'path-to-corpus-webis-editorial-quality-18.json'

        corpus = pd.read_json(path_corpus)

        return corpus

    def load_personality_traits(self):

        # Download the json fole for corpus-webis-editorial-quality-18_annotators-personality-traits.csv
        path_corpus = loader.get_data_root() + 'path-to corpus-webis-editorial-quality-18_annotators-personality-traits.csv'

        personality_df = pd.read_csv(path)

        return personality_df

    def load_data_with_features(self):
        if self.df_features is None:
            self.df_features = pd.read_json('data/articles_with_adu_liwc_lexicons.json', orient='records')
            self.df_features.set_index('idx', inplace=True)
        return self.df_features



    # helper method to get content
    @staticmethod
    def apply_get_article_content(row):
        article_id =  row['article_id']
        path = '../corpus/{}'.format(article_id)
        text = ''# get text
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
            row['content'] = text.strip()
            
        return row

    @staticmethod
    def apply_add_majority(row):
        effect_map = {0: 'challenging', 1: 'no_effect', 2:'reinforcing' }
        results = [row['challenging'], row['no_effect'],  row['reinforcing']]

        multi_majority = len([x for x in results if x == np.max(results)]) > 1
        effect_num = np.argmax(results) # if equal between challending an and non challengung, returns 0 which is considered challenging
        effect_num = 2 if  (multi_majority and effect_num==1) else effect_num ## we set it to reinforcing in case of tie with no_effect

        row['majority'] = effect_map[effect_num]
        row['majority_int'] = effect_num
        row['no_annotation'] =( np.max(results)) == 0
        row['multi_majority'] = multi_majority
        return row

    ## helper:
    @staticmethod
    def apply_add_ids(row):
        row['ids'] = int(row['article_id'].replace('.txt', ''))
        return row

    @staticmethod
    def apply_set_idx(row):
        row['idx'] =  int(row['ids'].split(',')[0]) if (',' in str(row['ids'])) else int(row['ids'])
        return row

    @staticmethod
    def sum_str(series):
       return reduce(lambda x, y: str(x) +','+ str(y), series)

    def get_article_dfs_per_ideology(self,  ideology = 'political_pole', include_content = False):
        #if self.data_division is None:
        #    print("train/test/validation set were not specified - Setting train test to 80\% 20\% ")
        #    self.get_train_test_data(train_percent = 0.8, has_validation_data = False, 
        #                                          add_to_corpus = True)
        result = {}
        for ideology, ideology_df in self.corpus.groupby([ideology]):
            df = pd.DataFrame(columns=['article_id', 'challenging', 'no_effect', 'reinforcing', 'split_label'])
            for aid, ideology_df in ideology_df.groupby(['article_id']):
                vals = ideology_df['effect_abstracted'].value_counts()
                row = {}
                row['article_id'] = aid
                for k in vals.keys():
                    row[self.abstract_effect_mapping[k]] = vals[k]
                
                #article_id_int = int((row['article_id'].split('.')[0]))
                #for k in self.data_division.keys():
                #    if article_id_int in self.data_division[k]:
                #        row['split_label'] = k
                #        break    
                df = df.append(row, ignore_index=True)
                df = df.fillna(0)
                
                if include_content:
                    #if self.pars_features_df is None:
                    #    self.load_paragraphs_with_features()
                    df = df.apply(loader.apply_get_article_content, axis=1)
            print("articles dataframe for ideology {} was created".format(ideology))
            print('The id of the df is the article id without txt')
            df = df.apply(loader.apply_add_ids, axis=1)
            df = df.groupby(['content'],as_index=False).agg({'challenging': 'sum',
                                                            'no_effect': 'sum',
                                                            'reinforcing': 'sum',
                                                            'ids':loader.sum_str})
            df = df.apply(loader.apply_set_idx, axis=1)
            df = df.apply(loader.apply_add_majority, axis=1)
            self.get_train_test_data(train_percent = 0.8, has_validation_data = False, add_to_corpus = True, 
            article_ids = list(df['idx'].values))

            df.set_index('idx', inplace=True)
            print('length of self.df: ', len(df))
            print('length of self.data_division_df: ', len(self.data_division_df))
            df = df.join(self.data_division_df)
            print('length of self.df: ', len(df))
            result[ideology] = df
        return result


    #def get_ideologies_hotencoded_effect():
    @staticmethod
    def apply_ideology_intensity(row):
        ideology = row['political_typology']
        
        if ideology in loader.lean_ideologies: 
            row['ideology_intensity'] = 'lean'
        elif ideology in loader.extreme_ideologies: 
            row['ideology_intensity'] = 'extreme'
        else:
            print(ideology, ' not specified')
        return row

    def add_ideology_intensity(self):
        print('"corpus" is set with ideology_intensity: "extreme" and "lean"')

        self.corpus = self.corpus.apply(loader.apply_ideology_intensity, axis = 1)
        return self.corpus



    def add_personality_label(self, personality_df, column_name='personality'): # the personality df is created independetly it has the annotators id and personality label
        print('"corpus" is set with personality: "a" , "b", ...')
        personality_df.columns = [column_name]
        self.corpus = self.corpus.merge(personality_df, how="inner", left_on='annotator_id', right_index=True )
        return self.corpus

    def get_train_test_data(self, train_percent = 0.8, has_validation_data = True, add_to_corpus=False, article_ids = None):
        article_ids = [int((x.split('.')[0])) for x in self.corpus['article_id'].unique()] if( article_ids is None) else article_ids
        
        # dismiss duplicates
        
        article_ids.sort()
        print('total con:', len(article_ids))
        # TRAINGING 
        training_num = round(len(article_ids)*train_percent)
        
        train_article_ids = article_ids[0: training_num ]

        print('rounded TRAINGING data: ', len(train_article_ids))


        # define validation and test size
        test_percent = ((1-train_percent)/2) if has_validation_data else 1-train_percent
        validate_percent = test_percent if has_validation_data else 0

        validate_article_ids = article_ids[training_num: training_num+round(len(article_ids)*validate_percent) ]
        test_article_ids = article_ids[-round(len(article_ids)*test_percent):]

        print('rounded Validation data: ', len(validate_article_ids))
        print('rounded Test data: ', len(test_article_ids))
        
 
        self.data_division = {
            "train": train_article_ids,
            "test": test_article_ids,
        }

        df_train = pd.DataFrame({'idx': train_article_ids, 'split_label': 'train'})
        df_test = pd.DataFrame({'idx': test_article_ids, 'split_label': 'test'})
        df = df_train.append(df_test)

        if has_validation_data:
            self.data_division["validate"]  = validate_article_ids
            df_validate = pd.DataFrame({'idx': test_article_ids, 'split_label': 'validation'})
            df = df.append(df_validate)
        
        
        if add_to_corpus:
            train_test_dict = self.get_train_test_data(train_percent =train_percent, has_validation_data = has_validation_data, add_to_corpus=False)
            print('"corpus" is set with split_label: "train" "test"')
            self.corpus = self.corpus.apply(loader.apply_test_train_split, args=(train_test_dict,),  axis=1)
        print('"data_division" is set as dict with keys ', self.data_division.keys())
        df.set_index('idx', inplace=True)
        self.data_division_df = df
        return self.data_division
    #def get_ideologies_hotencoded_effect():
    @staticmethod
    def apply_test_train_split(row, test_train_dict):
        article_id_int = int((row['article_id'].split('.')[0]))
        

        for k in test_train_dict.keys():
            if article_id_int in test_train_dict[k]:
                row['split_label'] = k
                break

        return row