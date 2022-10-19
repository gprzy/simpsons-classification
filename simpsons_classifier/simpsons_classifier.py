import sys
sys.path.append('..')

import numpy as np

from simpsons_classifier.voter import Voter

class SimpsonsClassifier():
    """
    Modelo de classificação dos personagens dos Simpsons
    """
    def __str__(self):
        return __class__.__name__ + f'({self.fieds})'
    
    def __init__(self, stack_models={}):
        # dicionário de (campos: stacks)
        self.stack_models = stack_models

        # lista de modelos (stacks) e campos
        self.models = list(self.stack_models.values())
        self.fields = list(self.stack_models.keys())

        self.voter = Voter()
        
        # dicionários de predições
        self.preds = {}
        self.preds_proba = {}

    def fit(self,
            X_train: list or np.array,
            y_train: list or np.array) -> None:
        """
        Realiza o fit do modelo nos dados, ou seja,
        o fit de cada stack nos respectivos dados
        """
        for stack, x_train in zip(self.models, X_train):
            stack.fit(x_train, y_train)

    def predict(self, X_test: list or np.array) -> dict:
        """
        Realiza as predições e retorna um dicionário com os
        valores preditos de cada modelo (stack)
        """
        for stack, field, x_test in zip(self.models, self.fields, X_test):
            y_pred = stack.predict(x_test)
            self.preds[field] = y_pred
        return self.preds

    def predict_proba(self, X_test: list or np.array):
        """
        Realiza as predições (por probabilidade) e
        retorna um dicionário com os valores preditos
        de cada modelo (stack)
        """
        for stack, field, x_test in zip(self.models, self.fields, X_test):
            y_pred = stack.predict_proba(x_test)
            self.preds_proba[field] = y_pred
        return self.preds_proba
