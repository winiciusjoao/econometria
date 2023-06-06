import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import seaborn as sns; sns.set()
import numpy as ny
import pandas_datareader.data as web
import matplotlib
matplotlib.rcParams['figure.figsize'] = (14,7)

def consulta_bc(codigo_bcb):
    url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
    df = pd.read_json(url)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df.set_index('data', inplace=True)
    return df

IPCA = consulta_bc(433)
IPCA
print(IPCA)
M1 = consulta_bc(27841)
M1
print(M1)
SELIC = consulta_bc(4390)
SELIC
print(SELIC)
PIB_M = consulta_bc(4380)
PIB_M
print(PIB_M)
demand = pd.DataFrame()
start_date = M1.index[0]
demand['M1'] = M1/M1.loc[start_date]
demand['PIB_M'] = PIB_M/PIB_M.loc[start_date]
demand['IPCA'] = IPCA/IPCA.loc[start_date]
demand['SELIC'] = SELIC/SELIC.loc[start_date]
demand
print(demand)

y = demand ['M1']
x1 = demand['SELIC']
x2 = demand['PIB_M']
x3 =demand['IPCA']

plt.scatter(x1,y)
plt.axis([0,2,0,10])
plt.xlabel('M1')
plt.ylabel('SELIC')
plt.show()

#regressão simples
X = sm.add_constant(x1)
reg = sm.OLS(y,X).fit()
reg.summary()
print(reg.summary())

#regressão multipla
x = np.column_stack((x1,x2,x3))
x = sm.add_constant(x)
reg2 = sm.OLS(y,x).fit()
reg2.summary()
print(reg2.summary())

#regressão em log
y = np.log(demand['M1'])
x1 = demand ['PIB_M']
x2 = demand ['IPCA']
x3 = demand ['SELIC']

x = np.column_stack((x1,x2,x3))
x = sm.add_constant(x)
reg3 = sm.OLS(y,x).fit()
reg3.summary()
print(reg3.summary())