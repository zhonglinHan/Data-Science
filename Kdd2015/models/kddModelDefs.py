
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from nolearn.dbn import DBN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

class Models:
       def __init__(self):
           self.modelname = []
           self.model_best = []
           self.model = []
           self.params_best = []
           self.params_cv = []
           self.feature_used = []
       
       def def_feature_all(self):
           self.feature_all = ['event_trend', 'events_last_week',
       'events_first_week', 'events_second_last_week', 'weekly_avg',
       'weekly_std', 'max_weekly_count', 'min_weekly_count',
       'first_event_month', 'last_event_month', 'session_count_3hr',
       'session_avg_3hr', 'session_std_3hr', 'session_max_3hr',
       'session_min_3hr', 'quadratic_b', 'quadratic_c',
       'session_count_1hr', 'session_avg_1hr', 'sessioin_std_1hr',
       'sessioin_max_1hr', 'session_min_1hr', 'session_dur_avg_3hr',
       'session_dur_std_3hr', 'sessioin_dur_max_3hr',
       'session_dur_min_3hr', 'sessioin_dur_avg_1hr',
       'session_dur_std_1hr', 'session_dur_max_1hr', 'session_dur_min_1hr',
       'MonCount', 'TueCount', 'WedCount', 'ThuCount', 'FriCount',
       'SatCount', 'SunCount', 'Hr0Count', 'Hr1Count', 'Hr2Count',
       'Hr3Count', 'Hr4Count', 'Hr5Count', 'Hr6Count', 'Hr7Count',
       'Hr8Count', 'Hr9Count', 'Hr10Count', 'Hr11Count', 'Hr12Count',
       'Hr13Count', 'Hr14Count', 'Hr15Count', 'Hr16Count', 'Hr17Count',
       'Hr18Count', 'Hr19Count', 'Hr20Count', 'Hr21Count', 'Hr22Count',
       'Hr23Count', 'AccCount', 'ProCount', 'PagCount', 'NagCount',
       'VidCount', 'DisCount', 'WikCount', 'BroCount', 'SerCount',
       'prob_ct', 'vid_ct', 'seq_ct', 'cha_ct', 'com_ct', 'ndate',
       'date_duration', 'study_days', 'npobj', 'nvobj', 'naobj', 'nwobj',
       'ndobj', 'nnobj', 'ncobj', 'std_events', 'skew_events',
       'kurt_events', 'std_obj', 'skew_obj', 'kurt_obj', 'std_ratio',
       'skew_ratio', 'kurt_ratio', 'module_total_num', 'about_num',
       'chapter_num', 'combinedopenended_num', 'course_num',
       'course_info_num', 'dictation_num', 'discussion_num', 'html_num',
       'outlink_num', 'peergrading_num', 'problem_num', 'sequential_num',
       'static_tab_num', 'vertical_num', 'video_num', 'father_num',
       'children_num', 'drop_rate', 'mean_hour', 'std_hour', 'skew_hour',
       'kurt_hour', 'score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30', 'acc_ob', 'pro_ob',
       'pag_ob', 'nav_ob', 'vid_ob', 'dis_ob', 'wik_ob', 'etp_1', 'etp_2',
       'etp_3', 'sum_last_6_sc', 'psum_last_6_sc', 'min_last_6_sc',
       'max_last_6_sc', 'pmax_last_6_sc', 'duration_per_date',
       'score_day_1_x',
       'score_day_2_x', 'score_day_3_x', 'score_day_4_x', 'score_day_5_x',
       'score_day_6_x', 'score_day_7_x', 'score_day_8_x', 'score_day_9_x',
       'score_day_10_x', 'score_day_11_x', 'score_day_12_x',
       'score_day_13_x', 'score_day_14_x', 'score_day_15_x',
       'score_day_16_x', 'score_day_17_x', 'score_day_18_x',
       'score_day_19_x', 'score_day_20_x', 'score_day_21_x',
       'score_day_22_x', 'score_day_23_x', 'score_day_24_x',
       'score_day_25_x', 'score_day_26_x', 'score_day_27_x',
       'score_day_28_x', 'score_day_29_x', 'score_day_30_x']


       def def_models(self, modelname):
           self.modelname = modelname
           if modelname == 'random_forest':
                self.def_random_forest()
           elif modelname == 'svm':
                self.def_svm()
           elif modelname == 'xgb':
                self.def_xgb()
           elif modelname == 'boost':
                self.def_boost()
           elif modelname == 'logistic':
                self.def_logistic()
           elif modelname == 'lda':
                self.def_lda()
           elif modelname == 'dbn':
                self.def_dbn()
           elif modelname == 'knn':
                self.def_knn()
           else:
               print "Model is Not Definied!"

       def def_random_forest(self):
           self.params_cv = {'max_features': [40],
                        'max_depth': [80],
                        'min_samples_split': [5],
                        'min_samples_leaf': [60],
                        'n_estimators': [1000]}
           self.params_best = {'max_features': 40,
                        'max_depth': 80,
                        'min_samples_split': 5,
                        'min_samples_leaf': 60,
                        'n_estimators': 2000}
           self.model = RandomForestClassifier()
           self.model_best = RandomForestClassifier(max_features = self.params_best['max_features'],
                                                    max_depth = self.params_best['max_depth'],
                                                    min_samples_split = self.params_best['min_samples_split'],
                                                    min_samples_leaf = self.params_best['min_samples_leaf'],
                                                    n_estimators=self.params_best['n_estimators'])
           
           self.feature_used = ['event_trend', 'events_last_week',
       'events_first_week', 'events_second_last_week', 'weekly_avg',
       'weekly_std', 'max_weekly_count', 'min_weekly_count',
       'first_event_month', 'last_event_month', 'session_count_3hr',
       'session_avg_3hr', 'session_std_3hr', 'session_max_3hr',
       'session_min_3hr', 'quadratic_b', 'quadratic_c',
       'session_count_1hr', 'session_avg_1hr', 'sessioin_std_1hr',
       'sessioin_max_1hr', 'session_min_1hr', 'session_dur_avg_3hr',
       'session_dur_std_3hr', 'sessioin_dur_max_3hr',
       'session_dur_min_3hr', 'sessioin_dur_avg_1hr',
       'session_dur_std_1hr', 'session_dur_max_1hr', 'session_dur_min_1hr',
       'MonCount', 'TueCount', 'WedCount', 'ThuCount', 'FriCount',
       'SatCount', 'SunCount', 'Hr0Count', 'Hr1Count', 'Hr2Count',
       'Hr3Count', 'Hr4Count', 'Hr5Count', 'Hr6Count', 'Hr7Count',
       'Hr8Count', 'Hr9Count', 'Hr10Count', 'Hr11Count', 'Hr12Count',
       'Hr13Count', 'Hr14Count', 'Hr15Count', 'Hr16Count', 'Hr17Count',
       'Hr18Count', 'Hr19Count', 'Hr20Count', 'Hr21Count', 'Hr22Count',
       'Hr23Count', 'AccCount', 'ProCount', 'PagCount', 'NagCount',
       'VidCount', 'DisCount', 'WikCount', 'BroCount', 'SerCount',
       'prob_ct', 'vid_ct', 'seq_ct', 'cha_ct', 'com_ct', 'ndate',
       'date_duration', 'study_days', 'npobj', 'nvobj', 'naobj', 'nwobj',
       'ndobj', 'nnobj', 'ncobj', 'std_events', 'skew_events',
       'kurt_events', 'std_obj', 'skew_obj', 'kurt_obj', 'std_ratio',
       'skew_ratio', 'kurt_ratio', 'module_total_num', 'about_num',
       'chapter_num', 'combinedopenended_num', 'course_num',
       'course_info_num', 'dictation_num', 'discussion_num', 'html_num',
       'outlink_num', 'peergrading_num', 'problem_num', 'sequential_num',
       'static_tab_num', 'vertical_num', 'video_num', 'father_num',
       'children_num', 'drop_rate', 'mean_hour', 'std_hour', 'skew_hour',
       'kurt_hour', 'score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30', 'acc_ob', 'pro_ob',
       'pag_ob', 'nav_ob', 'vid_ob', 'dis_ob', 'wik_ob', 'etp_1', 'etp_2',
       'etp_3']
       
       def def_svm(self):
           self.params_cv = {'C': [3,2,1],
                        'kernel': ['linear'],
                        'class_weight': [None,'auto']}
           self.params_best = {'C':1, 'kernel':'linear','class_weight': None}
           self.model_best = SVC(C = self.params_best['C'],
                                 kernel = self.params_best['kernel'],
                                 class_weight = self.params_best['class_weight'],
                                 probability = True,
                                 verbose = True)
           self.model = SVC()
           self.feature_used = ['ndate', 'date_duration', 'study_days',
                           'problem', 'npobj', 'video', 'nvobj',
                           'access', 'naobj', 'wiki', 'nwobj', 'discussion', 'ndobj',
                           'navigate', 'nnobj', 'page_close', 'ncobj','std_events','std_obj']
       
       def def_xgb(self):
           self.params_cv = {'objective':['binary:logistic'],
                        'learning_rate': [0.005],
                        'gamma': [0],
                        'max_depth': [6],
                        'min_child_weight': [1],
                        'max_delta_step': [1],
                        'subsample': [0.6],
                        'colsample_bytree':[0.6],
                        'n_estimators': [2000],
                        'nthread':[8]}
           self.params_best = {'objective':'binary:logistic',
                        'learning_rate': 0.005,
                        'gamma': 0,
                        'max_depth': 6,
                        'min_child_weight': 1,
                        'max_delta_step': 1,
                        'subsample': 0.6,
                        'colsample_bytree':0.6,
                        'n_estimators': 2000}
           self.model = xgb.XGBClassifier()
           self.model_best = xgb.XGBClassifier(objective = self.params_best['objective'],
                                               learning_rate = self.params_best['learning_rate'],
                                               gamma = self.params_best['gamma'],
                                               max_depth = self.params_best['max_depth'],
                                               min_child_weight = self.params_best['min_child_weight'],
                                               max_delta_step = self.params_best['max_delta_step'],
                                               subsample = self.params_best['subsample'],
                                               colsample_bytree = self.params_best['colsample_bytree'],
                                               n_estimators = self.params_best['n_estimators'],
                                               nthread = 8, silent = False)
           #self.features_ambg = ['about_num','cha_ct','chapter_num','children_num',
           #'com_ct','combinedopenended_num','course_info_num','course_num','dictation_num','first_event_month',
           #'html_num','outlink_num','peergrading_num','static_tab_num']
           
           self.feature_used = ['event_trend', 'events_last_week',
       'events_first_week', 'events_second_last_week', 'weekly_avg',
       'weekly_std', 'max_weekly_count', 'min_weekly_count',
       'first_event_month', 'last_event_month', 'session_count_3hr',
       'session_avg_3hr', 'session_std_3hr', 'session_max_3hr',
       'session_min_3hr', 'quadratic_b', 'quadratic_c',
       'session_count_1hr', 'session_avg_1hr', 'sessioin_std_1hr',
       'sessioin_max_1hr', 'session_min_1hr', 'session_dur_avg_3hr',
       'session_dur_std_3hr', 'sessioin_dur_max_3hr',
       'session_dur_min_3hr', 'sessioin_dur_avg_1hr',
       'session_dur_std_1hr', 'session_dur_max_1hr', 'session_dur_min_1hr',
       'MonCount', 'TueCount', 'WedCount', 'ThuCount', 'FriCount',
       'SatCount', 'SunCount', 'Hr0Count', 'Hr1Count', 'Hr2Count',
       'Hr3Count', 'Hr4Count', 'Hr5Count', 'Hr6Count', 'Hr7Count',
       'Hr8Count', 'Hr9Count', 'Hr10Count', 'Hr11Count', 'Hr12Count',
       'Hr13Count', 'Hr14Count', 'Hr15Count', 'Hr16Count', 'Hr17Count',
       'Hr18Count', 'Hr19Count', 'Hr20Count', 'Hr21Count', 'Hr22Count',
       'Hr23Count', 'AccCount', 'ProCount', 'PagCount', 'NagCount',
       'VidCount', 'DisCount', 'WikCount', 'BroCount', 'SerCount',
       'prob_ct', 'vid_ct', 'seq_ct', 'cha_ct', 'com_ct', 'ndate',
       'date_duration', 'study_days', 'npobj', 'nvobj', 'naobj', 'nwobj',
       'ndobj', 'nnobj', 'ncobj', 'std_events', 'skew_events',
       'kurt_events', 'std_obj', 'skew_obj', 'kurt_obj', 'std_ratio',
       'skew_ratio', 'kurt_ratio', 'module_total_num', 'about_num',
       'chapter_num', 'combinedopenended_num', 'course_num',
       'course_info_num', 'dictation_num', 'discussion_num', 'html_num',
       'outlink_num', 'peergrading_num', 'problem_num', 'sequential_num',
       'static_tab_num', 'vertical_num', 'video_num', 'father_num',
       'children_num', 'drop_rate', 'mean_hour', 'std_hour', 'skew_hour',
       'kurt_hour', 'score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30', 'acc_ob', 'pro_ob',
       'pag_ob', 'nav_ob', 'vid_ob', 'dis_ob', 'wik_ob', 'etp_1', 'etp_2',
       'etp_3', 'sum_last_6_sc', 'psum_last_6_sc', 'min_last_6_sc',
       'max_last_6_sc', 'pmax_last_6_sc', 'duration_per_date', 'min_lag',
       'max_lag', 'mean_lag', 'std_lag']
           self.feature_dict = dict(zip(self.feature_used,range(len(self.feature_used))))
           #self.feature_used = list(set(self.feature_used) - set(self.features_ambg))

       def def_boost(self):
       #max_leaf_nodes : int or None, optional (default=None)
#Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. If not None then max_depth will be ignored.
           self.params_cv = {'loss': ['deviance'],
                         'learning_rate': [0.01,0.005],
                         'n_estimators': [500,1000],
                         'subsample': [0.8, 1.0],
                         'min_samples_split': [2],
                         'min_samples_leaf': [1],
                         'min_weight_fraction_leaf': [0],
                         'max_depth': [4,6,8],
                         'max_features': [3,6]}
           self.params_best = {'loss': 'deviance',
                            'learning_rate': 0.01,
                            'n_estimators': 500,
                            'subsample': 0.6,
                            'min_samples_split': 2,
                            'min_samples_leaf': 1,
                            'min_weight_fraction_leaf': 0,
                            'max_depth': 6,
                            'max_features': 3}

           self.model = GradientBoostingClassifier()
           self.model_best = GradientBoostingClassifier(loss = self.params_best['loss'],
                                                        learning_rate = self.params_best['learning_rate'],
                                                        n_estimators = self.params_best['n_estimators'],
                                                        subsample = self.params_best['subsample'],
                                                        min_samples_split = self.params_best['min_samples_split'],
                                                        min_samples_leaf = self.params_best['min_samples_leaf'],
                                                        min_weight_fraction_leaf = self.params_best['min_weight_fraction_leaf'],
                                                        max_depth = self.params_best['max_depth'],max_features = self.params_best['max_features'])
           self.feature_used = ['ndate', 'date_duration', 'study_days',
                           'problem', 'npobj', 'video', 'nvobj',
                           'access', 'naobj', 'wiki', 'nwobj', 'discussion', 'ndobj','navigate', 'nnobj', 'page_close', 'ncobj','std_obj','std_events']
       def def_dbn(self):
           self.feature_used = ['event_trend', 'events_last_week',
       'events_first_week', 'events_second_last_week', 'weekly_avg',
       'weekly_std', 'max_weekly_count', 'min_weekly_count',
       'first_event_month', 'last_event_month', 'session_count_3hr',
       'session_avg_3hr', 'session_std_3hr', 'session_max_3hr',
       'session_min_3hr', 'quadratic_b', 'quadratic_c',
       'session_count_1hr', 'session_avg_1hr', 'sessioin_std_1hr',
       'sessioin_max_1hr', 'session_min_1hr', 'session_dur_avg_3hr',
       'session_dur_std_3hr', 'sessioin_dur_max_3hr',
       'session_dur_min_3hr', 'sessioin_dur_avg_1hr',
       'session_dur_std_1hr', 'session_dur_max_1hr', 'session_dur_min_1hr',
       'MonCount', 'TueCount', 'WedCount', 'ThuCount', 'FriCount',
       'SatCount', 'SunCount', 'Hr0Count', 'Hr1Count', 'Hr2Count',
       'Hr3Count', 'Hr4Count', 'Hr5Count', 'Hr6Count', 'Hr7Count',
       'Hr8Count', 'Hr9Count', 'Hr10Count', 'Hr11Count', 'Hr12Count',
       'Hr13Count', 'Hr14Count', 'Hr15Count', 'Hr16Count', 'Hr17Count',
       'Hr18Count', 'Hr19Count', 'Hr20Count', 'Hr21Count', 'Hr22Count',
       'Hr23Count', 'AccCount', 'ProCount', 'PagCount', 'NagCount',
       'VidCount', 'DisCount', 'WikCount', 'BroCount', 'SerCount',
       'prob_ct', 'vid_ct', 'seq_ct', 'cha_ct', 'com_ct', 'ndate',
       'date_duration', 'study_days', 'npobj', 'nvobj', 'naobj', 'nwobj',
       'ndobj', 'nnobj', 'ncobj', 'std_events', 'std_obj', 'mean_hour',
       'score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30']
           self.params_cv = {'layer_sizes' : [[len(self.feature_used),100,200,2]],
                             'scales' : [0.3],
                             'output_act_funct': ['Softmax'],
                             'use_re_lu': [False],
                             'learn_rates': [0.05],
                             'learn_rate_decays': [0.9],
                             'learn_rate_minimums':[0.001],
                             'momentum': [0.9],
                             'l2_costs': [0.0001],
                             'dropouts': [0,0.1],
                             'epochs': [200],
                             'minibatch_size': [64]}
           self.params_best = {'layer_sizes' : [len(self.feature_used),300,600,2],
                             'scales' : 1.0,
                             'output_act_funct': 'Softmax',
                             'use_re_lu': False,
                             'learn_rates': 0.01,
                             'learn_rate_decays': 0.9,
                             'learn_rate_minimums': 0.0001,
                             'momentum': 0.9,
                             'l2_costs': 0.0001,
                             'dropouts': 0.0,
                             'epochs': 200,
                             'minibatch_size': 64}
           self.model = DBN()
           self.model_best = DBN(layer_sizes = self.params_best['layer_sizes'],
                                 scales = self.params_best['scales'],
                                 output_act_funct=self.params_best['output_act_funct'],
                                 use_re_lu=self.params_best['use_re_lu'],
                                 learn_rates=self.params_best['learn_rates'],
                                 learn_rate_decays=self.params_best['learn_rate_decays'],
                                 learn_rate_minimums=self.params_best['learn_rate_minimums'],
                                 momentum=self.params_best['momentum'],
                                 l2_costs=self.params_best['l2_costs'],
                                 dropouts=self.params_best['dropouts'],
                                 epochs=self.params_best['epochs'],
                                 minibatch_size=self.params_best['minibatch_size'],verbose=1)

       def def_knn(self):
           self.params_cv = {'n_neighbors': [25,30,35],
                        'weights': ['uniform','distance'],
                        'algorithm': ['kd_tree'],
                        'leaf_size': [10,20,30],
                        'metric': ['minkowski'],
                        'p': [1, 2]}
           self.params_best = {'n_neighbors': 5,
                            'weights': 'uniform',
                            'algorithm': 'auto',
                            'leaf_size': 30,
                            'metric': 'minkowski',
                            'p': 2}
           self.model = KNeighborsClassifier()
           self.model_best = KNeighborsClassifier(n_neighbors = self.params_best['n_neighbors'],
                                                  weights = self.params_best['weights'],
                                                  algorithm = self.params_best['algorithm'],
                                                  leaf_size = self.params_best['leaf_size'],
                                                  metric = self.params_best['metric'],
                                                  p = self.params_best['p'])
           self.feature_used = ['ndate', 'date_duration', 'study_days',
                           'problem', 'npobj', 'video', 'nvobj',
                           'access', 'naobj', 'wiki', 'nwobj', 'discussion', 'ndobj',
                           'navigate', 'nnobj', 'page_close', 'ncobj','std_obj','std_events']

       def def_logistic(self):
           self.params_cv = {'penalty': ['l1'],
                        'tol': [0.0001],
                        'C': [0.001],
                        'class_weight': [None],
                        'solver': ['liblinear'],
                        'multi_class': ['ovr']}
           self.params_best = {'penalty':'l1',
                            'tol': 0.0001 ,
                            'C': 0.5,
                            'class_weight': None,
                            'solver': 'liblinear',
                            'multi_class': 'ovr'}
           self.model = LogisticRegression()
           self.model_best = LogisticRegression(penalty = self.params_best['penalty'],
                                                tol = self.params_best['tol'],
                                                C = self.params_best['C'],
                                                class_weight = self.params_best['class_weight'],
                                                solver = self.params_best['solver'],
                                                multi_class = self.params_best['multi_class'])
           self.feature_used = ['event_trend', 'events_last_week',
       'events_first_week', 'events_second_last_week', 'weekly_avg',
       'weekly_std', 'max_weekly_count', 'min_weekly_count',
       'first_event_month', 'last_event_month', 'session_count_3hr',
       'session_avg_3hr', 'session_std_3hr', 'session_max_3hr',
       'session_min_3hr', 'quadratic_b', 'quadratic_c',
       'session_count_1hr', 'session_avg_1hr', 'sessioin_std_1hr',
       'sessioin_max_1hr', 'session_min_1hr', 'session_dur_avg_3hr',
       'session_dur_std_3hr', 'sessioin_dur_max_3hr',
       'session_dur_min_3hr', 'sessioin_dur_avg_1hr',
       'session_dur_std_1hr', 'session_dur_max_1hr', 'session_dur_min_1hr',
       'MonCount', 'TueCount', 'WedCount', 'ThuCount', 'FriCount',
       'SatCount', 'SunCount', 'Hr0Count', 'Hr1Count', 'Hr2Count',
       'Hr3Count', 'Hr4Count', 'Hr5Count', 'Hr6Count', 'Hr7Count',
       'Hr8Count', 'Hr9Count', 'Hr10Count', 'Hr11Count', 'Hr12Count',
       'Hr13Count', 'Hr14Count', 'Hr15Count', 'Hr16Count', 'Hr17Count',
       'Hr18Count', 'Hr19Count', 'Hr20Count', 'Hr21Count', 'Hr22Count',
       'Hr23Count', 'AccCount', 'ProCount', 'PagCount', 'NagCount',
       'VidCount', 'DisCount', 'WikCount', 'BroCount', 'SerCount',
       'prob_ct', 'vid_ct', 'seq_ct', 'cha_ct', 'com_ct', 'ndate',
       'date_duration', 'study_days', 'npobj', 'nvobj', 'naobj', 'nwobj',
       'ndobj', 'nnobj', 'ncobj', 'std_events', 'skew_events',
       'kurt_events', 'std_obj', 'skew_obj', 'kurt_obj', 'std_ratio',
       'skew_ratio', 'kurt_ratio', 'module_total_num', 'about_num',
       'chapter_num', 'combinedopenended_num', 'course_num',
       'course_info_num', 'dictation_num', 'discussion_num', 'html_num',
       'outlink_num', 'peergrading_num', 'problem_num', 'sequential_num',
       'static_tab_num', 'vertical_num', 'video_num', 'father_num',
       'children_num', 'drop_rate', 'mean_hour', 'std_hour', 'skew_hour',
       'kurt_hour', 'score_day_1', 'score_day_2', 'score_day_3',
       'score_day_4', 'score_day_5', 'score_day_6', 'score_day_7',
       'score_day_8', 'score_day_9', 'score_day_10', 'score_day_11',
       'score_day_12', 'score_day_13', 'score_day_14', 'score_day_15',
       'score_day_16', 'score_day_17', 'score_day_18', 'score_day_19',
       'score_day_20', 'score_day_21', 'score_day_22', 'score_day_23',
       'score_day_24', 'score_day_25', 'score_day_26', 'score_day_27',
       'score_day_28', 'score_day_29', 'score_day_30']

       def def_lda(self):
           self.params_best = {'solver': 'svd',
                            'shrinkage': None,
                            'n_components': None,
                            'store_covariance': False,
                            'tol': 0.001}
           self.params_cv = {'solver': ['svd'],
                             'shrinkage': [None],
                             'n_components': [None],
                             'store_covariance': [False, True],
                             'tol': [0.001, 0.05]}
           self.model = LDA()
           self.model_best = LDA(solver = self.params_best['solver'],
                                 shrinkage = self.params_best['shrinkage'],
                                 n_components = self.params_best['n_components'],
                                 store_covariance = self.params_best['store_covariance'],
                                 tol = self.params_best['tol'])
           self.feature_used = ['ndate', 'date_duration', 'study_days',
                            'problem', 'npobj', 'video', 'nvobj',
                            'access', 'naobj', 'wiki', 'nwobj', 'discussion', 'ndobj',
                            'navigate', 'nnobj', 'page_close', 'ncobj','std_obj','std_events']






