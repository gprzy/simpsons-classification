from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import StackingClassifier

# estimadores finais de cada stacking classifier
final_estimators = [
    LGBMClassifier(random_state=42),
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

setup = (estimators, final_estimators)

def load_stacking_models(fields, setup=setup):
    estimators = setup[0]
    final_estimators = setup[1]
    
    # lista de stacks instanciadas
    stacks = {}
    
    for i in range(len(fields)):
        stack = StackingClassifier(estimators=estimators,
                                   final_estimator=final_estimators[i],
                                   stack_method='auto',
                                   verbose=0)

        # adicionando a stack ao dicionário
        stacks[fields[i]] = stack

    return stacks