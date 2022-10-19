import numpy as np
import pandas as pd

class Voter():
    @staticmethod
    def hard_voting(args: list or np.array,
                    weights: list or np.array = None) -> np.array:
        """
        Realiza uma votação com base em diferentes valores,
           retornando uma lista com os vereditos.
        
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