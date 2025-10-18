import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# load and process data
data = pd.read_csv('data.csv', header=None, usecols=[0, 1, 2], names=['genus', 'latitude', 'age'])
data = data[['genus', 'latitude', 'age']]

data['genus'] = data['genus'].astype(str)
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['age'] = data['age'].astype(str)

def run_anova(df):
    model = ols('latitude ~ C(age)', data=df).fit()
    return sm.stats.anova_lm(model, typ=2)

# Q1: one-way ANOVA to test latitude differences across age groups
spisula = data[data['genus'] == 'spisula']
print(run_anova(spisula), '\n')

genera = ['astarte',
          'chama',
          'chesapecten',
          'crepidula',
          'cyclocardia',
          'modiolus',
          'mulinia',
          'noetia',
          'parvilucina',
          'spisula',
          'striarca']

def run_tukey(df):
    return pairwise_tukeyhsd(endog=df['latitude'], groups=df['age'], alpha=0.05)

# Q2: post-hoc Tukey HSD test on genera across age groups
for genus in genera:
    print(f'Tukey HSD results for {genus}:')
    print(run_tukey(data[data['genus'] == genus]), '\n')
    
print(f'Tukey HSD results for all genera, ignoring age groups:')
print(pairwise_tukeyhsd(endog=data['latitude'], groups=data['genus'], alpha=0.05), '\n')
    
print(f'Tukey HSD results for all age groups, ignoring genera:')
print(pairwise_tukeyhsd(endog=data['latitude'], groups=data['age'], alpha=0.05), '\n')