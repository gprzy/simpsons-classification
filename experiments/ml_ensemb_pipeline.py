import argparse
import sys

sys.path.append('..')

import time
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_loader.load_data import ImagesLoader
from data_loader.colors import Colors

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, \
                             AdaBoostClassifier, \
                             ExtraTreesClassifier, \
                             StackingClassifier, \
                             VotingClassifier, \
                             BaggingClassifier

from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

plt.ioff()

parser = argparse.ArgumentParser(description='simpsons_classification')

parser.add_argument('--dataset', required=True)
parser.add_argument('--data_format', required=True)
parser.add_argument('--load_type', required=False, default='memory')

args = parser.parse_args()

DATASET_NAME = args.dataset
PIPELINE_NAME = args.data_format
LOAD_TYPE = args.load_type

def load_data():
    print(f'\n{Colors.WARNING}[loading data...]{Colors.ENDC}')

    loader = ImagesLoader(train_images_path=f'../data/{DATASET_NAME}/train/',
                          test_images_path=f'../data/{DATASET_NAME}/test/')

    # carregando os dados a partir da memória
    if LOAD_TYPE == 'memory':
        data = loader.load_data(load_list=[PIPELINE_NAME])

        print('data loaded from memory!')
    
    # carregando os dados a partir de um pickle em disco
    elif LOAD_TYPE == 'disk':
        with open(f'../data/{DATASET_NAME}/{DATASET_NAME}.pkl', 'rb') as infile:
            data = pickle.load(infile)

        print('data loaded from disk!')
        
    return data, loader

def show_data_info(data):
    print(f'\n{Colors.WARNING}[showing data shapes]{Colors.ENDC}')

    print('names_characters =',
          np.array(data['names_characters']['train']).shape,
          np.array(data['names_characters']['test']).shape)

    print(PIPELINE_NAME + ' =',
          np.array(data[PIPELINE_NAME]['train']).shape,
          np.array(data[PIPELINE_NAME]['test']).shape, end='\n\n')

def train_test_split(data):
    print(f'{Colors.WARNING}[train-test split]{Colors.ENDC}')

    # tornar dados 1d em caso de imagens
    if len(data[PIPELINE_NAME]['train'].shape) >= 3 or \
       len(data[PIPELINE_NAME]['test'].shape) >= 3:

        X_train = data[PIPELINE_NAME]['train'].reshape(data[PIPELINE_NAME]['train'].shape[0], -1)
        X_test = data[PIPELINE_NAME]['test'].reshape(data[PIPELINE_NAME]['test'].shape[0], -1)

    # caso não seja imagens, apenas carrega
    else:
        X_train = data[PIPELINE_NAME]['train']
        X_test = data[PIPELINE_NAME]['test']
    
    y_train = np.array(data['names_encoded']['train'])
    y_test = np.array(data['names_encoded']['test'])

    print('X_train shape =', X_train.shape)
    print('X_test shape =', X_test.shape)
    
    print('split finished!', end='\n\n')
    return X_train, y_train, X_test, y_test

def create_models():
    print(f'{Colors.WARNING}[creating models]{Colors.ENDC}')

    estimators = [
        ('lsvc', LinearSVC(random_state=42)),
        ('mlp', MLPClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42))
    ]

    # stacking classifiers com os estimators
    models = {
        f'stk_{estimators[i][0]}': StackingClassifier(
            estimators=estimators,
            final_estimator=estimators[i][1]
        ) for i in range(len(estimators))
    }

    # voting classifier com os estimators
    models['vote'] = VotingClassifier(estimators=estimators,
                                      voting='hard',
                                      weights=[1, 1, 1, 1, 1])

    # voting classifier com os estimators (weighted)
    models['vote_w'] = VotingClassifier(estimators=estimators,
                                      voting='hard',
                                      weights=[1, 1, 1, 1, 2])

    # voting classifiers com as melhores stacks
    models['vote_stk'] = VotingClassifier(
        estimators=[('stk_lgbm', models['stk_lgbm']),
                    ('stk_xgb', models['stk_xgb']),
                    ('stk_lsvc', models['stk_lsvc'])],
        voting='hard',
        weights=[1, 1, 1]
    )

    # stacking classifier de stacks (weighted)
    models['vote_stk_w'] = VotingClassifier(
        estimators=[('stk_lgbm', models['stk_lgbm']),
                    ('stk_xgb', models['stk_xgb']),
                    ('stk_lsvc', models['stk_lsvc'])],
        voting='hard',
        weights=[2, 1, 1]
    )

    # bag de lgbm
    models['bag_lgbm'] = BaggingClassifier(
        base_estimator=LGBMClassifier(random_state=42),
        n_estimators=20
    )

    # bag de xgb
    models['bag_xgb'] = BaggingClassifier(
        base_estimator=XGBClassifier(random_state=42),
        n_estimators=20
    )

    # bag de lsvc
    models['bag_lsvc'] = BaggingClassifier(
        base_estimator=LinearSVC(random_state=42),
        n_estimators=20
    )

    preds = {}
    results = {}

    print('models created!')
    return models, preds, results

def train_and_test(models,
                   preds,
                   results,
                   X_train,
                   y_train,
                   X_test,
                   y_test):
    # treinando e realizando as predições
    print(f'\n{Colors.WARNING}[training and testing models...]{Colors.ENDC}')

    for name, model in zip(list(models.keys()), list(models.values())):    
        pipe = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(0,1))),
            ('model', model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        preds[name] = y_pred
        results[name] = classification_report(y_test, y_pred, output_dict=True)
            
        print(f'{Colors.OKGREEN}*{Colors.ENDC}', name, 'completed')

    print('train and test completed!')

def save_model_reports(results, loader):
    # criando um df de resultados dos modelos
    print(f'\n{Colors.WARNING}[saving models classification reports]{Colors.ENDC}')

    for name, result, i in zip(list(results.keys()),
                               list(results.values()),
                               range(len(results))): 
        # df do classification report 
        df_report = pd.DataFrame(data=results[name]).T

        # dicionário para decodificar labels
        decoded_labels = {value: key for value, key in zip(list(loader.encoded_labels.values()),
                                                           list(loader.encoded_labels.keys()))}

        # decodificando labels
        df_report.index = list(map(lambda field: decoded_labels[int(field)] \
                                   if field in np.array(list(loader.encoded_labels.values())).astype(str) \
                                   else field, list(df_report.index)))

        # definindo index com o modelo
        arrays = [[name for i in range(9)], df_report.T.columns]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['model', 'labels'])
        df_report.index = index

        # mesclando os df_reports
        if i == 0:
            df_results = df_report
        else:
            df_results = pd.concat([df_results, df_report], axis=0)

    # salvar resultados
    df_results.to_csv(
        f'../output/{DATASET_NAME}/classification-report/pipeline_ensemb_{PIPELINE_NAME}_results.csv'
    )

    print('models reports saved!')

def save_models_confusion_matrices(results, preds, y_test, loader):
    print(f'\n{Colors.WARNING}[saving models confusion matrices]{Colors.ENDC}')

    fig, ax = plt.subplots(4,3, figsize=(16,20))

    fig.text(x=.5,
             y=.92,
             s=f'Confusion Matrix: pipeline "{PIPELINE_NAME}"\ndataset="{DATASET_NAME}"',
             horizontalalignment='center',
             verticalalignment='top',
             fontsize=16)

    for name, pred, i in zip(list(preds.keys()),
                             list(preds.values()),
                             range(len(preds))):
        
        cm = confusion_matrix(y_test, pred)
        cm_plot = ConfusionMatrixDisplay(cm, display_labels=loader.labels)
        cm_plot.plot(cmap='viridis_r', ax=ax.ravel()[i]);
        cm_plot.im_.colorbar.remove()

        ax.ravel()[i].set_title(f"{name}; accuracy = " \
                                f"{round(results[name]['accuracy'], 2)}; " \
                                f"weight f1 = " \
                                f"{round(results[name]['weighted avg']['f1-score'], 2)}");

    plt.savefig(
        f'../output/{DATASET_NAME}/confusion-matrix/pipeline_ensemb_{PIPELINE_NAME}_cm.jpg'
    )
    
    print('models confusion matrices saved!')

if __name__ == '__main__':
    start = time.time()

    try:
        print(f"{Colors.HEADER}[PROCESS STARTED]{Colors.ENDC}")
        print(f"dataset name = {Colors.OKBLUE}'{DATASET_NAME}'{Colors.ENDC}")
        print(f"pipeline name = {Colors.OKBLUE}'{PIPELINE_NAME}'{Colors.ENDC}")

        # load data
        data, loader = load_data()
        print(f'{Colors.OKCYAN}{round(time.time() - start, 4)}s elapsed{Colors.ENDC}')

        # data info
        show_data_info(data)

        # train-test split
        X_train, y_train, X_test, y_test = train_test_split(data)
        models, preds, results = create_models()
        
        # train and test models
        train_and_test(models=models,
                       preds=preds,
                       results=results,
                       X_train=X_train,
                       y_train=y_train,
                       X_test=X_test,
                       y_test=y_test)

        print(f'{Colors.OKCYAN}{round(time.time() - start, 4)}s elapsed{Colors.ENDC}')

        # saving results
        save_model_reports(results, loader)
        save_models_confusion_matrices(results, preds, y_test, loader)

        # salvando o log da execução
        with open(f'../logs/{DATASET_NAME}.log', 'a') as logfile:
            logfile.write(
                f'pipeline (ensembles) "{PIPELINE_NAME}" finished with SUCCESS; ' \
                f'time elapsed={round(time.time() - start, 4)}s\n'
            )

        print(f'\n{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} pipeline finished!')
        print(f'{Colors.OKCYAN}TOTAL TIME elapsed = {round(time.time() - start, 4)}s{Colors.ENDC}')

    except Exception as e:
        print(f'\n{Colors.FAIL}[ERROR]{Colors.ENDC} pipeline not finished!')
        print(f'{Colors.OKCYAN}TOTAL TIME elapsed = {round(time.time() - start, 4)}s{Colors.ENDC}')

        # salvando o log do erro
        with open(f'../logs/{DATASET_NAME}.log', 'a') as logfile:
            logfile.write(
                f'pipeline (ensembles) "{PIPELINE_NAME}" finished with ERROR; ' \
                f'time elapsed={round(time.time() - start, 4)}s; ' \
                f'TRACEBACK={e}\n'
            )