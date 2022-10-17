import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import StackingClassifier

def models_setup():
    # campos de cada stacking classifier
    fields = [
        'combination_hsv+hu',
        'combination_hsv+lbp+hu',
        'descriptor_hsv'
    ]

    # estimadores finais de cada stacking classifier
    final_estimators = [
        XGBClassifier(random_state=42),
        LinearSVC(random_state=42),
        LGBMClassifier(random_state=42)
    ]

    # estimadores de cada stacking classifier
    estimators = [
        ('lsvc', LinearSVC(random_state=42)),
        ('mlp', MLPClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42))
    ]

    return (fields, final_estimators, estimators)

def load_stacking_models(load_type='memory',
                         setup=models_setup()):
    if load_type=='memory':
        fields = setup[0]
        final_estimators = setup[1]
        estimators = setup[2]
        
        # lista de stacks instanciadas
        stacks = {}
        
        for i in range(len(fields)):
            stack = StackingClassifier(estimators=estimators,
                                       final_estimator=final_estimators[i],
                                       stack_method='auto',
                                       verbose=0)

            # adicionando a stack ao dicionário
            stacks[fields[i]] = stack

    elif load_type == 'disk':
        pass

    return stacks