import random

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

class Voter():
    @staticmethod
    def hard_voting(args: list or np.array,
                    weights: list or np.array = None) -> np.array:
        """
        Realiza uma votação com base em diferentes valores,
        retornando uma lista com os vereditos. A votação ocorre
        mediante a contagem das predições realizadas, acatando
        o voto da maioria.
        
        Params:
            args: lista com vetores de valores a serem votados;
            weights: pesos a serem aplicados na votação;
                     se os pesos forem um vetor (1D), aplica os pesos
                     para cada modelo, conforme índice do peso na lista
                     respectivamente ao índice das predições do modelo
                     em args; se os pesos forem uma matriz (2D),
                     aplica analogamente os pesos para diferentes
                     modelos em cada classe, conforme o índice da lista
                     de pesos na matriz;
        Returns:
            (np.array): vetor com vereditos das votações;
        """
        elections = []
        for candidates in zip(*args):
            candidates = list(candidates)
            if weights:
                # aplicando pesos

                # pesos dos modelos diferentes para cada classe
                if np.array(weights).ndim == 2:
                    wcandidates = []
                    for i in range(len(candidates)):
                        wcandidates.append([candidates[i] for k in range(weights[candidates[i]][i])])
                
                # pesos para os modelos em geral
                else:
                    wcandidates = []
                    for i in range(len(candidates)):
                        wcandidates.append([candidates[i] for k in range(weights[i])])

                wcandidates = [candidate for group in wcandidates \
                            for candidate in group]

                candidates = wcandidates

            # extraindo a moda
            candidates = pd.Series(candidates)
            election = candidates.mode()[0]

            elections.append(election)
        return np.array(elections)

    def soft_voting(args: list or np.array) -> np.array:
        elections = []
        for candidates in zip(*args):
            elections.append(np.argmax(candidates))
        return np.array(elections)

    @staticmethod
    def hard_voting_random_optimization(preds,
                                        y_test,
                                        n=10000,
                                        verbose=False):
        # pesos aleatórios; scores obtidos (f1)
        random_weights = []
        weighted_f1 = []

        # gerando 10000 matrizes aleatórias de pesos
        for i in range(n):
            weights = [[random.randint(1,3) for i in range(3)] for i in range(5)]

            # realizando a votação
            y_pred_election = Voter.hard_voting(args=list(preds.values()),
                                                weights=weights)

            weighted_f1.append(
                f1_score(y_test, y_pred_election, average='weighted')
            )

            random_weights.append(weights)

        # index do melhor score
        idx = np.argmax(weighted_f1)
        weights = random_weights[idx]

        if verbose:
            print('best weight f1 =', weighted_f1[idx])
            print('weights')
            print(weights)

        return weights, weighted_f1