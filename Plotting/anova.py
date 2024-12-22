import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Korrekte Daten für Accuracy
accuracy_data = {
    'Pruning Rate': [0, 0.2, 0.4, 0.6, 0.8],
    'connection_sparsity_sgd_10_epochs': [0.7, 0.6167, 0.615, 0.5572, 0.5434],
    'connection_sparsity_sgd_50_epochs': [0.7, 0.7238, 0.72276, 0.6921, 0.6359],
    'local_structured_adam_10_epochs': [0.7, 0.46234, 0.42572, 0.37184, 0.28936],
    'local_structured_adam_50_epochs': [0.7, 0.67, 0.64, 0.57, 0.45],
    'local_structured_sgd_10_epochs': [0.7, 0.59, 0.53, 0.45, 0.29],
    'local_structured_sgd_50_epochs': [0.7, 0.69, 0.637, 0.535, 0.325]
}

# Korrekte Daten für MIS und MIS Confidence
data = {
    'Optimizer': ['SGD', 'SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam', 'Adam', 'SGD', 'SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam', 'Adam', 'SGD', 'SGD', 'SGD', 'SGD', 'SGD', 'SGD', 'SGD', 'SGD'],
    'Epochs': [10, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 10, 10],
    'Method': ['Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Local Structured', 'Connection Sparsity', 'Connection Sparsity', 'Connection Sparsity', 'Connection Sparsity', 'Connection Sparsity', 'Connection Sparsity', 'Connection Sparsity', 'Connection Sparsity'],
    'Pruning Rate': [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8],
    'MIS': [0.720036, 0.728486, 0.747248, 0.777145, 0.751525, 0.744485, 0.770141, 0.784530, 0.715374, 0.726474, 0.750037, 0.794107, 0.740784, 0.733364, 0.750329, 0.784549, 0.711564, 0.711341, 0.720187, 0.728514, 0.716359, 0.719275, 0.731437, 0.736890],
    'MIS Confidence': [0.677047, 0.685017, 0.702323, 0.730448, 0.714053, 0.703176, 0.726773, 0.738197, 0.673034, 0.684253, 0.705554, 0.749621, 0.703319, 0.692476, 0.709577, 0.739509, 0.670764, 0.670914, 0.678927, 0.687622, 0.674753, 0.676454, 0.688974, 0.692454]
}

# Daten in DataFrame umwandeln
df = pd.DataFrame(data)

# Accuracy-Daten hinzufügen
def get_accuracy(row):
    method = row['Method'].lower().replace(' ', '_')
    epochs = row['Epochs']
    key = f"{method}_sgd_{epochs}_epochs"
    pruning_rate = row['Pruning Rate']
    return accuracy_data[key][accuracy_data['Pruning Rate'].index(pruning_rate)]

df['Accuracy'] = df.apply(get_accuracy, axis=1)

# ANOVA für MIS
formula_mis = 'MIS ~ C(Optimizer) + C(Epochs) + C(Method) + C(Pruning_Rate) + Accuracy'
model_mis = ols(formula_mis, data=df).fit()
anova_results_mis = anova_lm(model_mis)
print("ANOVA results for MIS:")
print(anova_results_mis)

# ANOVA für MIS Confidence
formula_conf = 'MIS_Confidence ~ C(Optimizer) + C(Epochs) + C(Method) + C(Pruning_Rate) + Accuracy'
model_conf = ols(formula_conf, data=df).fit()
anova_results_conf = anova_lm(model_conf)
print("\nANOVA results for MIS Confidence:")
print(anova_results_conf)
