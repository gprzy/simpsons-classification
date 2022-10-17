# script criado para renomear as imagens
# da base de dados do kaggle, tornando o
# nome dos arquivos no padrão dos demais
# conjuntos (ex: homer001.jpg)

import sys
import os

if __name__ == '__main__':
    # base_path = '../data/simpsons-large/'
    base_path = sys.argv[1]

    dirs_names = os.listdir('../data/simpsons-large')
    print('dirs names')
    print(dirs_names)

    bart_files = os.listdir('../data/simpsons-large/bart_simpson')
    homer_files = os.listdir('../data/simpsons-large/homer_simpson')
    lisa_files = os.listdir('../data/simpsons-large/lisa_simpson')
    maggie_files = os.listdir('../data/simpsons-large/maggie_simpson')
    marge_files = os.listdir('../data/simpsons-large/marge_simpson')

    dirs_files = [bart_files, homer_files,
                lisa_files, maggie_files,
                marge_files]

    print('bart files exemples')
    print(bart_files[:5])

    for dir_name, dir_files in zip(dirs_names, dirs_files):
        print(dir_name, '=', len(dir_files))

    print('renaming')
    # renomeando
    for dir_name, dir_files in zip(dirs_names, dirs_files):
        file_path = base_path + dir_name + '/'
        for file in dir_files:
            new_name = dir_name.split('_')[0] + file.split('_')[1]
            os.rename(file_path + file, file_path + new_name)

    print('process finished!')