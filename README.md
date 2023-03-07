# Simpsons Classification

<img src="https://github.com/gprzy/credit-scoring/blob/main/assets/puc.png" width="30%" height="30%"/>

<div align="center">
    <img src="./assets/homer.png" width=40% height=40%></img>
</div>

## Objetivo 🎯
Classificar os personagens dos Simpsons utilizando modelos de Machine Learning.

<div align="center">
    <table>
        <tr>
            <td>Personagem</td>
            <td>Classe</td>
        </tr>
        <tr>
            <td>Bart</td>
            <td>0</td>
        </tr>
        <tr>
            <td>Homer</td>
            <td>1</td>
        </tr>
        <tr>
            <td>Lisa</td>
            <td>2</td>
        </tr>
        <tr>
            <td>Marge</td>
            <td>3</td>
        </tr>
        <tr>
            <td>Maggie</td>
            <td>4</td>
        </tr>
    </table>
</div>

## Conjuntos de imagens utilizadas 📦
As imagens utilizadas em treino e teste dos modelos podem ser encontradas e baixadas publicamente. Todas as imagens utilizadas podem ser encontradas nesse mesmo [repositório](). Foram utilizados ao todo, para treinamento, dois conjuntos de dados (que são distintos mas possuem intersecções); são eles:
- `simpsons-small`: conjunto de imagens testado inicialmente;
- `simpsons-small-balanced`: possui todas as imagens do conjunto acima, porém com um adicional de algumas imagens obtidas do conjunto [The Simpsons Characters Data (Kaggle)](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset);

## Processamento das imagens 🖼
Os dados foram carregados utilizando a classe `ImagesLoader` do módulo `data_loader.load_data.py`, lendo as imagens de treino e teste a partir do caminho especificado. Os dados são armazenados em um dicionário (`dict`) que contém o nome dos arquivos, nomes dos personagens, caminhos dos arquivos, imagens originais, redimensionadas, descritores das imagens, entre outras variações. Trata-se de uma compilação de todos os dados a serem testados em diferentes modelos.

Exemplos de descritores utilizados e outros componentes das imagens:
- Local Binary Patterns (LBP);
- Hu moments;
- Gabor;
- HOG;
- Histogramas RGB;
- Histogramas HSV;
- Combinações de dois ou mais itens;

Segue abaixo um exemplo da aplicação dos descritores nas imagens:

<div align="center">
    <img src="./assets/descriptors.jpg"></img>
</div>

## Pipelines  de Machine Learning ⚙
A escolha dos dados que serão utilizados por cada modelo não foi completamente aleatória. Foram executadas várias *pipelines* de modelos, para cada um dos campos de dados carregados (exceto os dados das imagens em si, apenas descritores, histogramas e combinações). Em cada um dos modelos testados, foi utilizado o objeto `Pipeline` do `sklearn`, criando uma *pipeline* com o modelo em si e um `StandardScaler`, normalizando os dados entre 0 e 1, e então conduzindo o processo (treino e teste). Além do mais, cada *pipeline* utilizou os seguintes modelos:

- `KNeighborsClassifier()`
- `GaussianNB()`
- `LinearSVC()`
- `SVC()`
- `LogisticRegression()`
- `RandomForestClassifier()`
- `LGBMClassifier()`
- `XGBClassifier()`
- `AdaBoostClassifier()`
- `ExtraTreesClassifier()`
- `MLPClassifier()`
- `DummyClassifier(strategy='stratified')`
- `StackingClassifier()`
- `VotingClassifier()`
- `BaggingClassifier()`

Segue abaixo um diagrama ilustrando os passos das *pipelines*:

<div align="center">
    <image src="./assets/diagrama_pipeline.jpg" width="70%"></image>
</div>

### Executando uma pipeline
```bash
cd ./experiments
```

```bash
python ./ml_pipeline.py --dataset '<NOME_DO_DATASET>' --data_format 'CAMPO_DO_DICIONÁRIO' --load_type '[disk|memory]'
```

O exemplo abaixo executa uma *pipeline* utilizando o *dataset* `simpsons-small-balanced` e o campo `descriptor_rgb+hsv+lbp`, carregando os dados do disco, através do arquivo `pickle`.

```bash
python ./ml_pipeline.py --dataset 'simpsons-small-balanced' --data_format 'descriptor_rgb+hsv+lbp' --load_type 'disk'
```

A execução da *pipeline* será salva em `log`, no diretório `./logs/<NOME_DO_DATASET>.log`.

### Exemplos de `log` gerado

Segue abaixo o exemplo de um `log` gerado, após a conclusão (com ou sem erros) da *pipeline* descrita acima:

```bash
pipeline "descriptor_blue" finished with SUCCESS; time elapsed=40.4574s
pipeline "descriptor_green" finished with SUCCESS; time elapsed=41.5765s
pipeline "descriptor_red" finished with SUCCESS; time elapsed=41.0123s
```

### Passo a passo da execução da pipeline

A execução das pipelines descrevem o passo a passo de sua execução no terminal, segue abaixo um exemplo:

<div align="center">
    <img src="./assets/pipeline_terminal_example.jpg" width="50%" height="50%"></img>
</div>

## Métrica utilizada 📈
A métrica escolhida para avaliar os modelos foi o F1 *score*, mais precisamente a média ponderada entre os F1 *scores* (**weighted F1 score**) das diferentes classes.

$$F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$$

$$\overline{F1} = \frac{(F1_x \cdot |x|) + (F1_y \cdot |y|) + \dots}{x + y + \dots}$$

$$\overline{F1} = \left( \sum_{i=0}^{n}F1_{k_i} \cdot |k_i| \right) \cdot \left( \sum_{i=0}^{n} \frac{1}{k_i} \right), \ \ k=\{x, \ y, \ z, \dots\}$$

Onde
- $k$: Conjunto das classes;
- $x$: Classe $x$;
- $|x|$: Cardinalidade da classe $x$, ou seja, número de exemplos que a compõe;

## Estrutura do projeto 🌱

```
simpsons-classification
├─ challenge/*
├─ data/*
├─ data_loader/
│  ├─ image_descriptors.py
│  ├─ load_data.py
│  ├─ rename_images_names.py
│  └─ save_data.py
├─ utils/
│  ├─ colors.py
│  └─ plot_utils.py
├─ experiments/
│  ├─ ml_ensemb_pipeline.py
│  └─ ml_pipeline.py
├─ notebooks/*
├─ output/*
├─ simpsons-classification.ipynb
└─ simpsons_classifier/
   ├─ load_stacking_models.py
   ├─ simpsons_classifier.py
   └─ voter.py
```

Onde:

- `challenge/*`: desafio dos Simpsons, com arquivo que gera predições visando obter o maior **weighted f1-score**;
- `data/*`: imagens de treino e teste;
- `data_loader/`: carregamento de imagens, descritores, entre outros;
- `utils/`: funções utilitárias para *pretty print* e plotagens de gráficos e imagens;
- `experiments/`: *pipelines* de Machine Learning com vários modelos;
- `notebooks/*`: passo a passo e visualização de funções de módulos, *pipelines* e exemplos ilustrativos;
- `output/*`: *output* da execução das *pipelines*;
- `simpsons_classifier/`: modelo principal para classificação dos personagens dos Simpsons;
- `simpsons-classification.ipynb`: *notebook* principal, compilando os resultados e o passo a passo realizados durante o projeto;

## Criação do modelo 🛠
O modelo criado é composto por três `StackingClassifier` treinados com combinações diferentes de dados. Cada *stack* é composta pelos seguintes modelos:

- `LinearSVC`
- `MLPClassifier`
- `LogisticRegression`
- `XGBClassifier`
- `LGBMClassifier`

Esses `StackingClassifier` criados foram treinados, respectivamente, com os dados da combinação de histogramas e descritores, são eles: HSV+HU, HSV+HU+LBP e HSV. Por fim, as predições de cada *stack* foram unificadas através de um sistema ponderado de votações, com pesos atribuídos de maneira distinta entre as classes.

## Resultados obtidos 🎯

O modelo foi capaz de **acertar 83% das imagens**, apresentando um **avg weighted f1-score** de **0.83**. Segue abaixo a matriz de confusão com os acertos e erros do modelo:

<div align="center">
    <img src="./assets/cm.jpg" width="50%" height="50%"></img>
</div>

## The Simpsons Challenge 🏆

O desafio visa conseguir a maior pontuação para o **avg weighted f1-score**. O exemplo abaixo executa o *script* que gera o arquivo de predições, passando os arquivos `train.txt` e `test.txt` com os respectivos caminhos das imagens de treino e teste de entrada, salvando as predições em `output.txt`.

```bash
cd ./challenge
```

```bash
python ./simpsons.py --train 'train.txt' --test 'test.txt' --output 'output.txt'
```
