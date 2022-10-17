import numpy as np
import pandas as pd

class Voter():
    @staticmethod
    def hard_voting(args, weights=None):
        elections = []
        for candidates in zip(*args):
            candidates = list(candidates)
            if weights:
                # aplicando pesos

                if np.array(weights).ndim == 2:
                    wcandidates = []
                    for i in range(len(candidates)):
                        wcandidates.append([candidates[i] for k in range(weights[candidates[i]][i])])
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