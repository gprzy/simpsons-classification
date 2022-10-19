# script utilizado para o desafio
# lê o caminho das imagens de treino e teste
# a partir de arquivos txt (--train e --test),
# cria e treina os modelos, salvando as
# predições em outro arquivo txt (--output)

import argparse
import time
import re
import sys

sys.path.append('..')

import numpy as np

from sklearn.metrics import classification_report

from data_loader.load_data import ImagesLoader
from utils.colors import Colors

from simpsons_classifier.voter import Voter as vote
import simpsons_classifier.load_stacking_models as stack
from simpsons_classifier.simpsons_classifier import SimpsonsClassifier

import warnings
warnings.filterwarnings('ignore')

classes = {
    'bart': 0,
    'homer': 1,
    'lisa': 2,
    'marge': 3,
    'maggie': 4
}

# carregando o nome das imagens de treino
def load_train_data(path):
    X_train_files = []
    y_train = []

    # abrindo caminhos das imagens de treino
    with open(path) as fd_train:
        for line in fd_train:
            line = line.strip()
            X_train_files.append(line)

            # removendo números e extensão (ex: 000.bmp)
            label = line.split('/')[-1]
            label = re.findall('[a-z]*', label)[0]

            label = classes[label]
            y_train.append(label)

        X_train_files = np.array(X_train_files)
        y_train = np.array(y_train)

    return X_train_files, y_train

# carregando o nome das imagens de teste
def load_test_data(path):
    X_test_files = []
    y_test = []

    with open(path) as fd_test:
        for line in fd_test:
            line = line.strip()
            X_test_files.append(line)	

            # removendo números e extensão (ex: 000.bmp)
            label = line.split('/')[-1]
            label = re.findall('[a-z]*', label)[0]

            label = classes[label]
            y_test.append(label)

        X_test_files = np.array(X_test_files)
        y_test = np.array(y_test)

    return X_test_files, y_test

# write output
def write_output(x, y, output_name):
    fd = open(output_name, 'w')

    for fname, label in zip(x, y):        
        fd.write(f'{fname} {label}\n')

    fd.close()

def load_images_data(dataset_name, fields):
    loader = ImagesLoader(train_images_path=f'../data/{dataset_name}/train/',
                          test_images_path=f'../data/{dataset_name}/test/')

    data = loader.load_data(load_list=fields)
    return data

def show_data_info(data, model_fields):
    print(f'\n{Colors.WARNING}' \
          f'[exibindo shape dos dados carregados (X_train/X_test)]' \
          f'{Colors.ENDC}')

    for field in model_fields:
        print(field + ' =',
              data[field]['train'].shape,
              data[field]['test'].shape)

def load_train_test_data(data, fields):
    X_train = [data[field]['train'] for field in fields]
    X_test = [data[field]['test'] for field in fields]

    return X_train, X_test

if __name__ == '__main__':
    print(sys.version)
    
    parser = argparse.ArgumentParser(description='simpsons_classification')

    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    DATASET_NAME = 'simpsons-small-balanced'
    
    LOAD_FIELDS = [
        'images_hsv',
        'images_h',
        'images_s',
        'images_v',
        'descriptor_h',
        'descriptor_s',
        'descriptor_v',
        'descriptor_hsv',
        'descriptor_hu',
        'descriptor_lbp',
        'combination_hsv+hu',
        'combination_hsv+lbp+hu',
    ]

    MODEL_FIELDS = [
        'combination_hsv+hu',
        'descriptor_hsv'
    ]

    start = time.time()
    print(f'{Colors.HEADER}[PROCESSO INICIADO]{Colors.ENDC}')

    # your code goes here
    # read image data based on train path samples
    print(f'\n{Colors.WARNING}' \
          f'[carregando o nome das imagens de treino e teste...]' \
          f'{Colors.ENDC}')

    # labels das imagens de treino e teste
    X_train_files, y_train = load_train_data(args.train)
    X_test_files, y_test = load_test_data(args.test)

    print('nome dos arquivos carregados!')
    print('shape dos dados (y_train/y_test) =',
           y_train.shape, y_test.shape)

    print('shape dos dados (X_train_files/X_test_files) =',
           X_train_files.shape, X_test_files.shape)

    print(f'\n{Colors.WARNING}[carregando os descritores e ' \
          f'dados das imagens...]{Colors.ENDC}')

    # process and feature extraction
    data = load_images_data(dataset_name=DATASET_NAME,
                            fields=LOAD_FIELDS)

    print('dados das imagens carregados!')
    show_data_info(data, model_fields=MODEL_FIELDS)

    # dados de treino e teste
    X_train, X_test = load_train_test_data(data, fields=MODEL_FIELDS)

    # training and evaluation
    print(f'\n{Colors.WARNING}[instanciando o modelo...]{Colors.ENDC}')
    
    print('carregando setup dos modelos...')
    stacks = stack.load_stacking_models()

    print('instanciando o modelo SimpsonsClassifier')
    simp = SimpsonsClassifier(stack_models=stacks)
    
    print('modelo instanciado')

    print(f'\n{Colors.WARNING}[realizando treinamento...]{Colors.ENDC}')
    simp.fit(X_train, y_train)
    print('treinamento finalizado!')

    preds = simp.predict(X_test)

    print(f'\n{Colors.WARNING}[realizando uma votação com as predições]{Colors.ENDC}')
    
    # votação
    weights = [[1, 1, 1],
               [1, 2, 1],
               [1, 3, 2],
               [2, 2, 3],
               [3, 2, 3]]

    y_pred = vote.hard_voting(list(preds.values()),
                              weights=weights)

    print('votação concluída!\n')
        
    # a dummy prediction (random)
    # to better illustrate, lets say that random function
    # could predict test samples; so, the 'y' is an array
    # of the same size as X_test. Both arrays are direct related

    # y_pred = np.random.randint(0, 6, size=X_test.shape[0])
    print(classification_report(y_test, y_pred))

    # writes the result to the output
    # X_test_files == test filenames
    # y_pred == predicted labels

    print(f'{Colors.WARNING}[salvando as predições...]{Colors.ENDC}')
    write_output(X_test_files, y_pred, args.output)
    print('predições salvas!')

    print(f'\n{Colors.OKGREEN}[SUCESSO]{Colors.ENDC} processo finalizado!')
    print(f'{Colors.OKCYAN}tempo total={round(time.time() - start, 4)}s{Colors.ENDC}')
    print(f'{Colors.OKCYAN}tempo total~{round((time.time() - start)/60, 2)}min{Colors.ENDC}')