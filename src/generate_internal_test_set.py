import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv("../data/ML-CUP22-TR.csv", header=None, sep=",")
test = pd.read_csv("../data/ML-CUP22-TS.csv", header=None, sep=",")

x_train = train.to_numpy().astype(np.float64)
x_test_blind = test.to_numpy().astype(np.float64)

print("Full train set shape: ", x_train.shape)
print("Blind test shape: ", x_test_blind.shape)


int_test = train.sample(frac=0.2, random_state=42)
int_train = train.drop(int_test.index)


print("Internal train set shape: ", int_test.to_numpy().shape)
print("Internal test set shape: ", int_train.to_numpy().shape)

int_train.to_csv("../data/ML-CUP22-INTERNAL-TR.csv", header=False, index=False)
int_test.to_csv("../data/ML-CUP22-INTERNAL-TS.csv", header=False, index=False)
