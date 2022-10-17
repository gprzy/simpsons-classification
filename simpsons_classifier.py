from voter import Voter

class SimpsonsClassifier():
    def __init__(self, stack_models={}):
        self.stack_models = stack_models

        self.models = list(self.stack_models.values())
        self.fields = list(self.stack_models.keys())
        self.voter = Voter()
        
        # predições
        self.preds = {}
        self.preds_proba = {}

    def fit(self, X_train, y_train):
        for stack, x_train in zip(self.models, X_train):
            stack.fit(x_train, y_train)

    def predict(self, X_test):
        for stack, field, x_test in zip(self.models, self.fields, X_test):
            y_pred = stack.predict(x_test)
            self.preds[field] = y_pred
        return self.preds

    def predict_proba(self, X_test):
        for stack, field, x_test in zip(self.models, self.fields, X_test):
            y_pred = stack.predict_proba(x_test)
            self.preds_proba[field] = y_pred
        return self.preds_proba
