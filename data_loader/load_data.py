# módulo destinado a carregar os dados das imagens,
# sejam nomes dos arquivos, features de descritores,
# imagens de descritores, histogramas RGB, entre outros

import os
import re

import numpy as np

import cv2

from data_loader.image_descriptors import ImageDescriptors as desc
from data_loader.colors import Colors

class ImagesLoader():
    def __init__(self, train_images_path, test_images_path):
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path

        # (campo: método)
        self.load_methods = {
            'names_files': self.load_names_files,
            'names_paths': self.load_names_paths,
            'names_characters': self.load_names_characters,
            'names_encoded': self.load_names_encoded,
            'images_original': self.load_images_original,
            'images_resized': self.load_images_resized,
            'images_gray': self.load_images_gray,
            'images_blue': self.load_images_blue,
            'images_green': self.load_images_green,
            'images_red': self.load_images_red,
            'images_hsv': self.load_images_hsv,
            'images_h': self.load_images_h,
            'images_s': self.load_images_s,
            'images_v': self.load_images_v,
            'images_lbp': self.load_descriptor_lbp,
            'images_hu': self.load_descriptor_hu_moments,
            'images_gabor': self.load_descriptor_gabor,
            'images_hog': self.load_descriptor_hog,
            'descriptor_blue': self.load_descriptor_blue,
            'descriptor_green': self.load_descriptor_green,
            'descriptor_red': self.load_descriptor_red,
            'descriptor_rgb': self.load_descriptor_rgb,
            'descriptor_h': self.load_descriptor_h,
            'descriptor_s': self.load_descriptor_s,
            'descriptor_v': self.load_descriptor_v,
            'descriptor_hsv': self.load_descriptor_hsv,
            'descriptor_lbp': self.load_descriptor_lbp,
            'descriptor_hu': self.load_descriptor_hu_moments,
            'descriptor_gabor': self.load_descriptor_gabor,
            'descriptor_hog': self.load_descriptor_hog,
            'combination_rgb+hsv': self.load_descriptor_combinations,
            'combination_rgb+hsv+lbp': self.load_descriptor_combinations,
            'combination_rgb+hsv+hu': self.load_descriptor_combinations,
            'combination_rgb+hsv+hog': self.load_descriptor_combinations,
            'combination_rgb+hsv+lbp+hu': self.load_descriptor_combinations,
            'combination_rgb+hsv+lbp+hog': self.load_descriptor_combinations,
            'combination_rgb+hsv+hu+hog': self.load_descriptor_combinations,
            'combination_rgb+hsv+lbp+hu+hog': self.load_descriptor_combinations,
            'combination_rgb+lbp': self.load_descriptor_combinations,
            'combination_rgb+hu': self.load_descriptor_combinations,
            'combination_rgb+hog': self.load_descriptor_combinations,
            'combination_rgb+lbp+hu': self.load_descriptor_combinations,
            'combination_rgb+lbp+hog': self.load_descriptor_combinations,
            'combination_rgb+hu+hog': self.load_descriptor_combinations,
            'combination_rgb+lbp+hu+hog': self.load_descriptor_combinations,
            'combination_hsv+lbp': self.load_descriptor_combinations,
            'combination_hsv+hu': self.load_descriptor_combinations,
            'combination_hsv+hog': self.load_descriptor_combinations,
            'combination_hsv+lbp+hu': self.load_descriptor_combinations,
            'combination_hsv+lbp+hog': self.load_descriptor_combinations,
            'combination_hsv+hu+hog': self.load_descriptor_combinations,
            'combination_hsv+lbp+hu+hog': self.load_descriptor_combinations
        }

        # dicionário de dados das imagens com (campo: dados)
        self.data = {field: {
            'train': [],
            'test': []
            } for field in list(self.load_methods.keys())
        }

        # classes
        self.labels = ['bart', 'homer', 'lisa', 'marge', 'maggie']
    
        # classes codificadas
        self.encoded_labels = {
            'bart': 0,
            'homer': 1,
            'lisa': 2,
            'marge': 3,
            'maggie': 4,
            'family': 5,
        }

        # classes decodificadas
        self.decoded_labels = {value: key for value, key in zip(list(self.encoded_labels.values()),
                                                                list(self.encoded_labels.keys()))}

    # carregando o nomes dos arquivos das imagens
    def load_names_files(self):
        self.data['names_files']['train'] = np.array(os.listdir(self.train_images_path))
        self.data['names_files']['test'] = np.array(os.listdir(self.test_images_path))
        return True

    # carregando o nomes dos caminhos + arquivos das imagens
    def load_names_paths(self):
        for field, path in zip(['train', 'test'],
                               [self.train_images_path, self.test_images_path]):
            self.data['names_paths'][field] = np.array([
                path + i \
                for i in self.data['names_files'][field]
            ])
        return True

    # carregando a str do nome dos personagens de cada imagem
    def load_names_characters(self):
        for field in ['train', 'test']:
            self.data['names_characters'][field] = np.array([
                re.findall('[a-z]*', i.split('.')[0])[0] \
                for i in self.data['names_files'][field]
            ])
        return True

    # encodando cada nome para um número
    def load_names_encoded(self):
        for field in ['train', 'test']:
            self.data['names_encoded'][field] = np.array([
                self.encoded_labels[i] \
                for i in self.data['names_characters'][field]
            ])
        return True

    # carregando um array com as imagens originais
    def load_images_original(self):
        for field in ['train', 'test']:
            self.data['images_original'][field] = np.array([
                cv2.imread(i) \
                for i in self.data['names_paths'][field]
            ])
        return True

    # carregando imagens originais redimensionadas
    def load_images_resized(self):
        # largura das imagens de treino e teste
        imgs_train_x = np.array(
            [img.shape[0] for img in np.hstack([self.data['images_original']['train'],
                                                self.data['images_original']['test']])
        ])

        # alturas das imagens de treino e teste
        imgs_train_y = np.array(
            [img.shape[1] for img in np.hstack([self.data['images_original']['train'],
                                                self.data['images_original']['test']])
        ])

        # médias de largura e altura
        x_axis_mean = int(round(imgs_train_x.mean(), 0))
        y_axis_mean = int(round(imgs_train_y.mean(), 0))

        # resize das imagens para (387, 309)
        for field in ['train', 'test']:
            self.data['images_resized'][field] = np.array([
                cv2.resize(img, (x_axis_mean, y_axis_mean)) \
                for img in self.data['images_original'][field]
            ])
        return True

    # array de imagens em escala de cinza
    def load_images_gray(self):
        for field in ['train', 'test']:
            self.data['images_gray'][field] = np.array([
                desc.image_to_gray(img) \
                for img in self.data['images_resized'][field]
            ])
        return True

    # carregando canal azul das imagens originais
    def load_images_blue(self):
        for field in ['train', 'test']:
            self.data['images_blue'][field] = np.array(
                [img[:,:,0] for img in self.data['images_resized'][field]]
            )
        return True

    # carregando canal verde das imagens originais
    def load_images_green(self):
        for field in ['train', 'test']:
            self.data['images_green'][field] = np.array(
                [img[:,:,1] for img in self.data['images_resized'][field]]
            )
        return True

    # carregando canal vermelho das imagens originais    
    def load_images_red(self):
        for field in ['train', 'test']:
            self.data['images_red'][field] = np.array(
                [img[:,:,2] for img in self.data['images_resized'][field]]
            )
        return True

    # histograma do canal azul
    def load_descriptor_blue(self):
        for field in ['train', 'test']:
            self.data['descriptor_blue'][field] = np.array(
                [desc.calc_hist(img) \
                 for img in self.data['images_blue'][field]]
            )
        return True

    # histograma do canal verde
    def load_descriptor_green(self):
        for field in ['train', 'test']:
            self.data['descriptor_green'][field] = np.array(
                [desc.calc_hist(img) \
                 for img in self.data['images_green'][field]]
            )
        return True

    # histograma do canal vermelho
    def load_descriptor_red(self):
        for field in ['train', 'test']:
            self.data['descriptor_red'][field] = np.array(
                [desc.calc_hist(img) \
                 for img in self.data['images_red'][field]]
            )
        return True

    # concatena os histogramas de b, g, r
    def load_descriptor_rgb(self):
        for field in ['train', 'test']:
            self.data['descriptor_rgb'][field] = np.array([
                np.hstack([blue, green, red]) \
                for blue, green, red in zip(self.data['descriptor_blue'][field],
                                            self.data['descriptor_green'][field],
                                            self.data['descriptor_red'][field])
                ])
        return True

    # carrega as imagens em hsv
    def load_images_hsv(self):
        for field in ['train', 'test']:
            self.data['images_hsv'][field] = np.array(
                [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) \
                 for img in self.data['images_resized'][field]]
            )
        return True

    # carrega o canal h das imagens em hsv
    def load_images_h(self):
        for field in ['train', 'test']:
            self.data['images_h'][field] = np.array(
                [img[:,:,0] for img in self.data['images_hsv'][field]]
            )
        return True

    # carrega o canal s das imagens em hsv
    def load_images_s(self):
        for field in ['train', 'test']:
            self.data['images_s'][field] = np.array(
                [img[:,:,1] for img in self.data['images_hsv'][field]]
            )
        return True

    # carrega o canal v das imagens em hsv
    def load_images_v(self):
        for field in ['train', 'test']:
            self.data['images_v'][field] = np.array(
                [img[:,:,2] for img in self.data['images_hsv'][field]]
            )
        return True

    # histograma do canal h
    def load_descriptor_h(self):
        for field in ['train', 'test']:
            self.data['descriptor_h'][field] = np.array(
                [desc.calc_hist(img) \
                 for img in self.data['images_h'][field]]
            )
        return True

    # histograma do canal s
    def load_descriptor_s(self):
        for field in ['train', 'test']:
            self.data['descriptor_s'][field] = np.array(
                [desc.calc_hist(img) \
                 for img in self.data['images_s'][field]]
            )
        return True

    # histograma do canal v
    def load_descriptor_v(self):
        for field in ['train', 'test']:
            self.data['descriptor_v'][field] = np.array(
                [desc.calc_hist(img) \
                 for img in self.data['images_v'][field]]
            )
        return True

    # concatena os histogramas de h, s e v
    def load_descriptor_hsv(self):
        for field in ['train', 'test']:
            self.data['descriptor_hsv'][field] = np.array([
                np.hstack([h, s, v]) \
                for h, s, v in zip(self.data['descriptor_h'][field],
                                   self.data['descriptor_s'][field],
                                   self.data['descriptor_v'][field])
                ])
        return True

    # descritor lbp
    def load_descriptor_lbp(self):

        if len(self.data['images_lbp']['train']) == 0:

            # imagens
            for field in ['train', 'test']:
                self.data['images_lbp'][field] = np.array(
                    [desc.lbp(img)[1] for img in \
                     self.data['images_gray'][field]]
                )

            # lbp hist
            for field in ['train', 'test']:
                self.data['descriptor_lbp'][field] = np.array(
                    [desc.lbp(img)[0] for img in \
                     self.data['images_gray'][field]]
                )
        return True

    # descriptor hu moments
    def load_descriptor_hu_moments(self):
        if len(self.data['images_hu']['train']) == 0:
            # imagens
            for field in ['train', 'test']:
                self.data['images_hu'][field] = np.array(
                    [desc.hu_moments(img)[1] for img in \
                     self.data['images_gray'][field]]
                )

            # hu moments
            for field in ['train', 'test']:
                self.data['descriptor_hu'][field] = np.array(
                    [desc.hu_moments(img)[0] for img in \
                     self.data['images_gray'][field]]
                )
        return True

    # descriptor gabor
    def load_descriptor_gabor(self):
        if len(self.data['images_gabor']['train']) == 0:
            # imagens
            for field in ['train', 'test']:
                self.data['images_gabor'][field] = np.array(
                    [desc.gabor(img)[1] for img in \
                     self.data['images_gray'][field]]
                )
            
            # gabor filters
            for field in ['train', 'test']:
                self.data['descriptor_gabor'][field] = np.array(
                    [desc.gabor(img)[0] for img in \
                     self.data['images_gray'][field]]
                )
        return True

    # descriptor hog
    def load_descriptor_hog(self):
        if len(self.data['images_hog']['train']) == 0:
            # imagens
            for field in ['train', 'test']:
                self.data['images_hog'][field] = np.array(
                    [desc.hog(img)[1] for img in \
                     self.data['images_gray'][field]]
                )
            
            # hog features
            for field in ['train', 'test']:
                self.data['descriptor_hog'][field] = np.array(
                    [desc.hog(img)[0] for img in \
                     self.data['images_gray'][field]]
                )
        return True

    # combinações de descritores
    def load_descriptor_combination(self, *descriptors):
        for field in ['train', 'test']:
            # nome do campo a partir dos descritores desejados
            comb_field = f"combination_{[desc_name + '+' for desc_name in descriptors]}" \
                                        .replace('[','') \
                                        .replace(']','') \
                                        .replace(',', '') \
                                        .replace(' ', '') \
                                        .replace("'", '')[:-1]

            # juntando os dados
            self.data[comb_field][field] = np.array([
                np.hstack(desc_data) for desc_data in list(
                    zip(*[self.data[f'descriptor_{desc_name}'][field] \
                          for desc_name in descriptors])
                )
            ])
        return True

    # método que chama o combinador para diferentes campos
    def load_descriptor_combinations(self):
        # combinações com rgb + hsv
        self.load_descriptor_combination('rgb', 'hsv')
        self.load_descriptor_combination('rgb', 'hsv', 'lbp')
        self.load_descriptor_combination('rgb', 'hsv', 'hu')
        self.load_descriptor_combination('rgb', 'hsv', 'hog')
        self.load_descriptor_combination('rgb', 'hsv', 'lbp', 'hu')
        self.load_descriptor_combination('rgb', 'hsv', 'lbp', 'hog')
        self.load_descriptor_combination('rgb', 'hsv', 'hu', 'hog')
        self.load_descriptor_combination('rgb', 'hsv', 'lbp', 'hu', 'hog')

        # combinações com rgb
        self.load_descriptor_combination('rgb', 'lbp')
        self.load_descriptor_combination('rgb', 'hu')
        self.load_descriptor_combination('rgb', 'hog')
        self.load_descriptor_combination('rgb', 'lbp', 'hu')
        self.load_descriptor_combination('rgb', 'lbp', 'hog')
        self.load_descriptor_combination('rgb', 'hu', 'hog')
        self.load_descriptor_combination('rgb', 'lbp', 'hu', 'hog')

        # combinações com hsv
        self.load_descriptor_combination('hsv', 'lbp')
        self.load_descriptor_combination('hsv', 'hu')
        self.load_descriptor_combination('hsv', 'hog')
        self.load_descriptor_combination('hsv', 'lbp', 'hu')
        self.load_descriptor_combination('hsv', 'lbp', 'hog')
        self.load_descriptor_combination('hsv', 'hu', 'hog')
        self.load_descriptor_combination('hsv', 'lbp', 'hu', 'hog')
        return True

    # método principal de carregamento
    def load_data(self, load_list=None):
        for field, load_method, i in zip(list(self.load_methods.keys()),
                                         list(self.load_methods.values()),
                                         range(len(self.load_methods))):

            # carregando apenas as imagens originais
            if load_list == 'original':
                if i <= 5:
                    load = load_method()
                    if load:
                        print(f"{Colors.OKGREEN}+{Colors.ENDC} '{field}' loaded")
                    else:
                        print(f'error on loading {field}')
                else:
                    break

            # se não tiver nenhuma lista ou tiver combinações, carrega tudo
            elif not load_list or 'combination' in load_list:
                load = load_method()

                if load:
                    print(f"{Colors.OKGREEN}+{Colors.ENDC} '{field}' loaded")
                else:
                    print(f"error on loading '{field}'")

            # selecionando apenas os descritores desejados
            elif load_list != None:
                # até o index 6, são carregadas as imagens originais
                if i > 6:
                    if field in load_list:
                        load = load_method()
                        
                        if load:
                            print(f"{Colors.OKGREEN}+{Colors.ENDC} '{field}' loaded")
                        else:
                            print(f'error on loading {field}')
                    else:
                        print(f"{Colors.FAIL}-{Colors.ENDC} '{field}' not in load list")
                else:
                    load = load_method()
                    if load:
                        print(f"{Colors.OKGREEN}+{Colors.ENDC} '{field}' loaded")
                    else:
                        print(f"error on loading '{field}'")

            else:
                print('opção inválida')
        
        return self.data

    