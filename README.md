# Simpsons Classification
👶 Simpsons character classification using Machine Learning and image processing.

<div align="center">
    <img src="./assets/homer.png" width=40% height=40%></img>
</div>

## Objetivo
Classificar os personagens dos Simpsons utilizando modelos de Machine Learning.

## Descritores de Imagens
<div align="center">
    <img src="./assets/descriptors.jpg"></img>
</div>

## Processamento das Imagens
Os dados foram carregados utilizando a classe `ImagesLoader` do módulo `data_loader.load_data.py`, lendo as imagens de treino e teste a partir do caminho especificado. Os dados são armazenados em um dicionário (`dict`) que contém o nome dos arquivos, nomes dos personagens, caminhos dos arquivos, imagens originais, redimensionadas, descritores das imagens, entre outras variações. Trata-se de uma compilação de todos os dados a serem testados em diferentes modelos.

Exemplos de descritores utilizados e outros componentes das imagens:
- *Local Binary Patterns* (LBP);
- *Hu moments*;
- *Gabor filters*;
- HOG;
- Histogramas RGB;
- Histogramas HSV;

## *Pipelines*
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

## Executando uma *pipeline*
```bash
cd ./experiments
```

```bash
python .\ml_pipeline.py --dataset '<NOME_DO_DATASET>' --data_format 'CAMPO_DO_DICIONÁRIO' --load_type '[disk|memory]'
```

O exemplo abaixo executa uma *pipeline* utilizando o *dataset* `simpsons-small-balanced` e o campo `descriptor_rgb+hsv+lbp`, carregando os dados do disco, através do arquivo `pickle`.

```bash
python .\ml_pipeline.py --dataset 'simpsons-small-balanced' --data_format 'descriptor_rgb+hsv+lbp' --load_type 'disk'
```

A execução da *pipeline* será salva em `log`, no diretório `./logs/<NOME_DO_DATASET>.log`.

## Exemplos de `log` gerado

Segue abaixo o exemplo de um `log` gerado, após a conclusão (com ou sem erros) da *pipeline* descrita acima:

```bash
pipeline "descriptor_blue" finished with SUCCESS; time elapsed=40.4574s
pipeline "descriptor_green" finished with SUCCESS; time elapsed=41.5765s
pipeline "descriptor_red" finished with SUCCESS; time elapsed=41.0123s
```

## Métrica escolhida
A métrica escolhida para avaliar os modelos foi o F1 *score*, mais precisamente a média ponderada entre os F1 *scores* (**weighted F1**) das diferentes classes.

$$F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$$
$$\overline{F1} = \frac{(F1_x \cdot |x|) + (F1_y \cdot |y|) + \dots}{x + y + \dots}$$
$$\overline{F1} = \left( \sum_{i=0}^{n}F1_{k_i} \cdot |k_i| \right) \cdot \left( \sum_{i=0}^{n} \frac{1}{k_i} \right), \ \ k=\{x, \ y, \ z, \dots\}$$

Onde
- $k$: Conjunto das classes;
- $x$: Classe $x$;
- $|x|$: Cardinalidade da classe $x$, ou seja, número de exemplos que a compõe;

## Modelo criado

## Executando o *script* para o desafio

O exemplo abaixo executa o *script* passando o arquivo `train.txt` com os caminhos das imagens de treino entrada, o arquivo `test.txt` com os caminhos das imagens de teste de entrada, e salva as predições em `predictions.txt`.

```bash
cd ./challenge
```

```bash
python .\simpsons.py --train 'train.txt' --test 'test.txt' --output 'predictions.txt'
```
