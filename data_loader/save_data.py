# script destinado a ler os dados das imagens
# em memória e salvar um arquivo pickle serializado
# em disco, passando o caminho dos dados

import argparse
import sys
import pickle

sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

from data_loader.load_data import ImagesLoader

parser = argparse.ArgumentParser(description='simpsons_classification')
parser.add_argument('--dataset', required=True)
parser.add_argument('--save_type', required=False, default='grouped')
args = parser.parse_args()

DATASET_NAME = args.dataset
SAVE_TYPE = args.save_type

if __name__ == '__main__':
    print('carregando dados...\n')
    loader = ImagesLoader(train_images_path=f'../data/{DATASET_NAME}/train/',
                          test_images_path=f'../data/{DATASET_NAME}/test/')

    if SAVE_TYPE == 'grouped':
        data = loader.load_data()
        print('\ndados carregados!')

        print('salvando os dados no formato pickle...')

        # salvando os dados no formato pickle
        with open(f'../data/{DATASET_NAME}/{DATASET_NAME}.pkl', 'wb') as outfile:
            pickle.dump(data, outfile)

        print('processo finalizado com sucesso!')

    elif SAVE_TYPE == 'splitted':
        # TODO a implementar modo fragmentado de salvar
        # arquivos grandes
        pass