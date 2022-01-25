import pandas as pd
import numpy as np

df = pd.read_csv("spn/data/Export_Textiles/Export_Textiles.tsv", sep="\t")

frame = {"Profit":np.float64}

df = df.astype(frame)

print(df)

df.to_csv("Export_Textiles.tsv",index=False,sep="\t")
