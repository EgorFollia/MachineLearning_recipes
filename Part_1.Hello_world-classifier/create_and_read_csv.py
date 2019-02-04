import csv, os
import pandas as pd

file_txt = os.getcwd() + '/fruits'
file_csv = os.getcwd() + '/fruits.csv'
file_xlsx = os.getcwd() + '/fruits.xlsx'

data = [['Weight', 'Texture', 'Label'],
        [140, 1, 0],
        [130, 1, 0],
        [150, 0, 1],
        [170, 0, 1]]

with open(file_txt, 'w+') as open_txt:
    writer = csv.writer(open_txt, delimiter = ',', quoting = csv.QUOTE_MINIMAL)
    writer.writerows(data)

df = pd.read_csv(file_txt)
df.to_csv(file_csv)
