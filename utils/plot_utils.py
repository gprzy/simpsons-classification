import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from data_loader.load_data import ImagesLoader

# função para exibir f1 scores em conjunto de uma coluna desejada
def plot_scores_by_col(col, df, figsize=(14,10)):
    fig, ax = plt.subplots(1,2, figsize=figsize);

    sns.barplot(
        data=df,
        x='f1-score',
        y=col,
        palette='viridis',
        ax=ax[0]
    );

    df_agg = df.groupby(col).agg('max')
    df_agg = df_agg.sort_values(by='f1-score', ascending=False)

    sns.barplot(
        data=df_agg,
        x='f1-score',
        y=df_agg.index,
        palette='viridis',
        ax=ax[1]
    );

    plt.tight_layout();
    ax[0].set_title(f'f1-scores médios por {col}');
    ax[1].set_title(f'f1-scores máximos por {col}');

# função para plotar erros/acertos do modelo
def plot_cases(y_test,
               y_pred,
               images,
               case='erros',
               figsize=(18,13)):
    # lista de classificações com acertos/erros
    elements = []
    for pred, true, img in zip(y_pred,
                               y_test,
                               images):
        if case == 'acertos':
            if pred == true:
                elements.append((img, true, pred))

        elif case == 'erros':
            if pred != true:
                elements.append((img, true, pred))

    root = np.sqrt(len(elements))
    size = int(root)

    fig, ax = plt.subplots(size, size+2, figsize=figsize)

    # título
    fig.text(x=.5,
             y=.92,
             s=f"{case.capitalize()} do modelo; total de {case} = {len(elements)} de " \
               f"{len(images)} amostras",
             horizontalalignment='center',
             verticalalignment='top',
             fontsize=16)

    # exibindo os acertos/erros
    for pack in enumerate(elements):
        try:
            i = pack[0]
            img = pack[1][0]
            true = pack[1][1]
            pred = pack[1][2]

            ax.ravel()[i].imshow(img, cmap='binary');
            ax.ravel()[i].set_title(f'true={ImagesLoader.decoded_labels[true]}; ' \
                                    f'pred={ImagesLoader.decoded_labels[pred]}');
        except:
            pass