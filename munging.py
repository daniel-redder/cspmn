import pandas as pd
import numpy as np

df = pd.read_csv("spn/data/CrossingTraffic/Computer_Diagnostician.tsv", sep="\t")

frame = {"Rework_Cost":np.float64}

df = df.astype(frame)

print(df)

df.to_csv("Computer_Diagnostician.tsv",index=False,sep="\t")