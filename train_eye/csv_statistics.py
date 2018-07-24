import random

import pandas as pd
import matplotlib.pyplot as plt
from trainer.cnn import number_of_samples_from_image_coors

def should_take_row(x):
    resample_n = number_of_samples_from_image_coors(x['x'],x['y'])
    print(resample_n)
    if random.randint(0, 4) < resample_n:
        return True

if __name__ == "__main__":
    df = pd.read_csv("indexes_landmarks.csv")
    print(df.describe())
    filtered = df[df.apply(should_take_row, axis=1) == True]
    filtered.to_csv("filtered.csv")
    print(filtered.describe())

    filtered.plot.scatter(x='x', y='y')
    plt.show()
    print(df.describe())
    bins = list(range(-1920, 1919, 100))

    binned = pd.cut(df['x'], bins=bins).value_counts()
    threshold = 0.5
    mask = binned > threshold
    tail_prob = binned.loc[~mask].sum()
    prob = binned.loc[mask]
    prob.plot(kind='bar')
    plt.xticks(rotation=25)
    plt.show()


    only_x_pos = filtered[filtered.x>0]
    only_x_neg = filtered[filtered.x<=0]

    x = filtered.x
    y = filtered.x

    print(only_x_neg.describe())
    print(only_x_pos.describe())
