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
from data_loader.colors import Colors
from simpsons_classifier import SimpsonsClassifier
from voter import Voter as vote
import load_stacking_models as stack

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simpsons_classification')

    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    start = time.time()
    print(f'{Colors.HEADER}[PROCESSO INICIADO]{Colors.ENDC}')

    # your code goes here
    # read image data based on train path samples
    print(f'\n{Colors.WARNING}[carregando o nome das imagens de treino e teste...]{Colors.ENDC}')

    # labels das imagens de treino e teste
    X_train_files, y_train = load_train_data(args.train)
    X_test_files, y_test = load_test_data(args.test)

    print('nome dos arquivos carregados!')
    print('shape dos dados (y_train/y_test) =', y_train.shape, y_test.shape)
    print('shape dos dados (X_train_files/X_test_files) =', X_train_files.shape, X_test_files.shape)

    print(f'\n{Colors.WARNING}[carregando os descritores e dados das imagens...]{Colors.ENDC}')

    # process and feature extraction
    # dados de treino e teste
    DATASET_NAME = 'simpsons-small-balanced'
    
    FIELDS = [
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

    loader = ImagesLoader(train_images_path=f'../data/{DATASET_NAME}/train/',
                          test_images_path=f'../data/{DATASET_NAME}/test/')

    data = loader.load_data(load_list=FIELDS)

    print('dados das imagens carregados!')
    print(f'\n{Colors.WARNING}[exibindo shape dos dados carregados (X_train/X_test)]{Colors.ENDC}')

    for field in ['descriptor_lbp',
                  'combination_hsv+hu',
                  'combination_hsv+lbp+hu']:
        print(field + ' =',
              data[field]['train'].shape,
              data[field]['test'].shape)

    X_train = [
        data['combination_hsv+hu']['train'],
        data['combination_hsv+lbp+hu']['train'],
        data['descriptor_hsv']['train'],
    ]

    X_test = [
        data['combination_hsv+hu']['test'],
        data['combination_hsv+lbp+hu']['test'],
        data['descriptor_hsv']['test'],
    ]

    y_train = data['names_encoded']['train']
    y_test = data['names_encoded']['test']

    # training and evaluation
    print(f'\n{Colors.WARNING}[instanciando o modelo...]{Colors.ENDC}')
    
    print('carregando setup dos modelos...')
    stacks = stack.load_stacking_models(load_type='memory')

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
               [1, 1, 1],
               [1, 1, 1],
               [3, 3, 3],
               [4, 4, 4]]

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