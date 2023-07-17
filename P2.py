import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tools.tools as smt
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning # omite avisos de warning no Python
import statsmodels.stats.diagnostic as smd # fornece os testes diagnósticos p/ heterocedasticidade
import matplotlib.pyplot as plt
import yfinance as yf


# Função para obter os preços das ações
def get_stock_data(ticker, start_date):
    stock = yf.download(ticker, start=start_date, end=None)
    return stock['Close']


# Definição dos tickers das ações e data de início
tickers = ['EMBR3.SA', 'GGBR3.SA', 'ITSA3.SA', '^BVSP']
start_date = '2005-01-01'

# Obtendo os preços das ações
data = pd.DataFrame()
for ticker in tickers:
    data[ticker] = get_stock_data(ticker, start_date)

# Normalizando os preços das ações
start_price = data.iloc[0]
normalized_data = data / start_price

# Calculando os retornos percentuais diários das ações
returns = normalized_data.pct_change().dropna()

# Scatter plot dos retornos em relação ao IBOVESPA
fig, ax = plt.subplots()
ax.scatter(returns['^BVSP'], returns['EMBR3.SA'], label='EMBR3')
ax.set_xlabel('IBOVESPA')
ax.set_ylabel('Returns')
ax.legend()
plt.show()
#GGBR3 Graph
fig, ax = plt.subplots()
ax.scatter(returns['^BVSP'], returns['GGBR3.SA'], label='GGBR3')
ax.set_xlabel('IBOVESPA')
ax.set_ylabel('Returns')
ax.legend()
plt.show()
#ITSA3 Graph
fig, ax = plt.subplots()
ax.scatter(returns['^BVSP'], returns['ITSA3.SA'], label='ITSA3')
ax.set_xlabel('IBOVESPA')
ax.set_ylabel('Returns')
ax.legend()
plt.show()

# Heterocedasticidade - Breusch-Pagan test
X = returns[['EMBR3.SA','GGBR3.SA', 'ITSA3.SA']]
X = sm.add_constant(X)
y = returns['^BVSP']
model = sm.OLS(y, X).fit()
residuals = model.resid
het_test = smd.het_breuschpagan(residuals, exog_het=X)
print('Breusch-Pagan Test:')
print('LM Statistic:', het_test[0])
print('LM p-value:', het_test[1])

# Heterocedasticidade - White test (Cross Terms)
white_test = smd.het_white(residuals, exog=X)
print('\nWhite Test:')
print('LM Statistic:', white_test[0])
print('LM p-value:', white_test[1])

# Comparação de volatilidade em relação ao IBOVESPA
volatility_comparison = returns.std() / returns['^BVSP'].std()
print('\nVolatility Comparison:')
print(volatility_comparison)
