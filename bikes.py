

import pandas as pd

df = pd.read_csv('/content/Sales.csv')

df

df.shape

df.columns.values

df1 = df.drop(columns = ['Day'])

df1 = df.drop(columns = ['Month'])

df1 = df.drop(columns = ['Year'], inplace=True)

df1.head()

import pandas as pd

df3 = pd.read_excel('/content/Bicicletas.xlsx')

df3

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

cols = df3.columns[:30] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df3[cols].isnull(), cmap=sns.color_palette(colours))
plt.show()

for col in df3.columns:
    pct_missing = np.mean(df3[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

df3.columns.values

df3['State'].cat.add_categories(new_categories=['U'], inplace=True)
df3.fillna(value={'State': 'U'}, inplace=True)
print(df3.info())

df3.dtypes

df3[['Unit_Price']] = df3[['Unit_Price']].astype(float)

df3.dtypes

df3

missing_data = df3.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print('')

df3.columns.values

import numpy as np
df3.replace("?", np.nan, inplace = True)
df3.dropna(subset=['Date'], axis = 0, inplace= True)
df3.reset_index(drop = True, inplace = True)

df3.replace("?", np.nan, inplace = True)
df3.dropna(subset=['Customer_Age'], axis = 0, inplace= True)
df3.reset_index(drop = True, inplace = True)

















average_column_name = df3['Profit'].astype(float).mean(axis = 0)

df3.dtypes



df3.duplicated().sum()

df3.isnull().sum()

df3.drop_duplicates()

df3.duplicated('Date').sum()

df3.columns.values

df3.drop_duplicates('Date').sum()

df3.duplicated('Date').sum()

df3.drop_duplicates('Date').sum()

df3.drop_duplicates(df3.columns[~df3.columns.isin(['id'])

df





import pandas as pd

df5 = pd.read_excel('/content/Bicicletas.xlsx')

df5

df5.duplicated('Date').sum()

df5.isnull().sum()

df5.isnull().sum().sum()

Nulos = df5[df5.isnull().any(1)]

Nulos

df_sin_nan = df5.dropna(how='any')

df5.isnull().sum().sum()

df5.dropna

df5.isnull().sum().sum()

import numpy as np

df5.dropna









df5.isnull().sum().sum()

df5.head()



df5.columns.values

df5 = df5.dropna(subset=['Date', 'Customer_Age', 'Age_Group', 'Customer_Gender', 'Country',
       'State', 'Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Profit',
       'Cost', 'Revenue'])

df5.isnull().sum()

# Commented out IPython magic to ensure Python compatibility.
!pip uninstall pandas
!pip install "pandas==0.25.3"
# %pylab inline

import pandas

pandas.__version__

import pandas as pd

from pandas_profiling import ProfileReport

prof = ProfileReport(df5)

prof

import pandas as pd

df5 = pd.read_excel('/content/Bicicletas.xlsx')

df5

df5.columns.values

tab = pd.crosstab(index=df5["Customer_Age"],columns="frecuencia")
print(tab)

tab = pd.crosstab(index=df5["Age_Group"],columns="frecuencia")
print(tab)

tab = pd.crosstab(index=df5["Customer_Gender"],columns="frecuencia")
print(tab)

tab = pd.crosstab(index=df5["Country"],columns="frecuencia")
print(tab)

tab = pd.crosstab(index=df5["State"],columns="frecuencia")
print(tab)

tab = pd.crosstab(index=df5["Order_Quantity"],columns="frecuencia")
print(tab)



import pandas as pd

df5 = pd.read_excel('/content/Bicicletas.xlsx')





tab = pd.crosstab(index=df5["Age_Group"],columns="frecuencia")
print(tab)

import numpy as np
import matplotlib.pyplot as plt
 
grafica = df5
serie_2 = [30278, 487, 20154, 10584]
 
 
numero_de_grupos = len(serie_2)
indice_barras = np.arange(numero_de_grupos)
ancho_barras =0.35

plt.bar(indice_barras + ancho_barras, serie_2, width=ancho_barras, label='Ventas')
plt.legend(loc='best')
plt.xticks(indice_barras + ancho_barras, ('Adults', 'Seniors', 'Young Adults', 'Youth '))
 

plt.xlabel('Grupos')
plt.title('Ventas según los grupos')
 
plt.show()

tab = pd.crosstab(index=df5["Customer_Gender"],columns="frecuencia")
print(tab)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
serie_3 = [29811, 31692]


numero_de_grupos = len(serie_3)
indice_barras = np.arange(numero_de_grupos)
ancho_barras =0.35

plt.bar(indice_barras + ancho_barras, serie_3, width=ancho_barras, label='Ventas')
plt.legend(loc='best')
plt.xticks(indice_barras + ancho_barras, ('F','M'))

plt. xlabel('Sexo')
plt.title('Ventas según el sexo')

tab = pd.crosstab(index=df5["Country"],columns="frecuencia")
print(tab)

import numpy as np
import matplotlib.pyplot as plt
serie_3 = [11839, 9104, 5759, 5676, 7049, 22076]


numero_de_grupos = len(serie_3)
indice_barras = np.arange(numero_de_grupos)
ancho_barras =0.35

plt.bar(indice_barras + ancho_barras, serie_3, width=ancho_barras, label='Ventas')
plt.legend(loc='best')
plt.xticks(indice_barras + ancho_barras, ('Australia','Canada','France','Germany','United Kingdom', 'United States'))

plt.xlabel('Pais')
plt.title('Ventas según el país')

tab = pd.crosstab(index=df5["State"],columns="frecuencia")
print(tab)

df5.State.values

import numpy as np
import matplotlib.pyplot as plt
serie_3 = [11839, 9104, 5759, 5676, 7049, 22076]


numero_de_grupos = len(serie_3)
indice_barras = np.arange(numero_de_grupos)
ancho_barras =0.35

plt.bar(indice_barras + ancho_barras, serie_3, width=ancho_barras, label='Ventas')
plt.legend(loc='best')
plt.xticks(indice_barras + ancho_barras, ('Australia','Canada','France','Germany','United Kingdom', 'United States'))

plt.xlabel('Pais')
plt.title('Ventas según el país')

df5.head()

df5= df5.drop(['Date'], axis=1)

df5

df5.isnull().sum()

df5 = df5.dropna(subset=['Customer_Age', 'Age_Group', 'Customer_Gender', 'Country',
       'State', 'Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Profit',
       'Cost', 'Revenue'], inplace=True)

df5 = df5

print(df5)



df5 = pd.read_excel('/content/Bicicletas.xlsx')

df5

df5= df5.drop(['Date'], axis=1)

df5

df5.isnull().sum()



df5 = df5.dropna(),

df5.isnull().sum()

df5

df5.to_excel('Bicis_limpio.xlsx')

df5.columns.values

tab = pd.crosstab(index=df5["Age_Group"],columns="frecuencia")
print(tab)

df5 = df5.Age_Group.replace(('Adults', 'Seniors', 'Young Adults', 'Youth'), (4, 3, 2, 1), inplace = True)

df5

import pandas as pd
import math

df7 = pd.read_excel('/content/Bicis_limpio.xlsx')

df7.head()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = [12,6]

import warnings
warnings.filterwarnings('ignore')

df7.describe().T

def plot_data_count(df7, col, return_pct_share=True, hue=None, figsize=(12,6)):
    
    plt.figure(figsize=figsize)
    g = sns.countplot(data=df7, x=col, hue=hue)
    for rect in g.patches:
        h = rect.get_height()
        w = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        g.annotate(f"{h}", (x+w/2, h), ha='center', va='bottom', fontsize=12)
    g.spines['top'].set_visible(False)
    g.spines['right'].set_visible(False)
    g.spines['left'].set_visible(False)
    
    plt.show()
    
    if return_pct_share:
        print("\n")
        print("Percent share for each category:")
        print(df[col].value_counts(normalize=True)*100)

df7.columns.values

plot_data_count(df7, 'Unit_Price')

plot_data_count(df7, 'Unit_Cost')

plot_data_count(df7, 'Order_Quantity')

plot_data_count(df7, 'State')

plot_data_count(df7, 'Country')

plot_data_count(df7, 'Customer_Gender')

plot_data_count(df7, 'Customer_Age')

plot_data_count(df7, 'Age_Group')

df7.columns.values

continuous_features = ['Customer_Age', 
                       'Age_Group', 
                       'Customer_Gender', 
                       'Country', 
                       'State', 
                       'Order_Quantity',
                       'Unit_Cost',
                       'Unit_Price',
                       'Profit',
                       'Cost',
                       'Revenue']

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,20))
axs = np.ravel(axs)
for i, col in enumerate(continuous_features):
    plt.sca(axs[i])
    sns.histplot(data=df7, x=col, kde=True, line_kws={'lw':2, 'ls':'--'}, color='orange')

plt.suptitle("Distribution Plot of Continuous Features", fontsize=18, color='#05445E', va='bottom')
plt.tight_layout()
plt.show()

# Análisis Univariado con funpymodeling

!pip install funpymodeling

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from funpymodeling.exploratory import status

from funpymodeling.exploratory import profiling_num #Análisis de variables númericas

from funpymodeling.exploratory import freq_tbl #Análisis de variables categoricas



status(df7)

profiling_num(df7)

freq_tbl(df7)

day_freq=freq_tbl(df7['State'])

day_freq              #Frecuencias   #Porcentaje   #cumulative_perc

df7.columns.values





#Gráficas de Google

google_analytics_features = ['Customer_Age', 'Cost', 'Revenue']

# Bounce Rate vs Exit Rate

sns.lmplot(x="Customer_Age", y="Cost", data=df7, 
           scatter_kws={'alpha':0.3}, 
           line_kws={'color':'red', 'ls':'--'})
plt.show()







