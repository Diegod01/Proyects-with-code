

import pandas as pd

df = pd.read_html('https://es.wikipedia.org/wiki/Anexo:Gobernantes_de_Uruguay', parse_dates=True)

print(df[0].head(15))

df[0].to_csv('presidentes.csv')









