import pandas as pd
from scipy.io import arff
from pprint import pprint

# 1. Load the ARFF file
data, meta = arff.loadarff('speeddating.arff')

# 2. Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# 3. Clean byte strings (Optional but recommended)
# scipy often loads text data as bytes (e.g., b'nominal_value')
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode('utf-8')

# 4. View the result
#print(df.T.head())
print(df.shape)
for row in df.head(3).to_dict(orient="records"):
    pprint(row)
    print()

lowest_row = df.loc[df["attractive"].idxmin()]
print(lowest_row)
print(lowest_row['attractive'])
print(lowest_row['attractive_o'])
