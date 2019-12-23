
import os
import variables
import matplotlib.pyplot as plt
import numpy as np

def number_of_data():
    labels = [d for d in os.listdir(variables.train_path) if os.path.isdir(variables.train_path + d)]
    recordings = []
    for label in labels:
        data = [f for f in os.listdir(variables.train_path + label) if f.endswith('.wav')]
        recordings.append(len(data))
    plt.figure(figsize=(30,5))
    index = np.arange(len(labels))
    plt.bar(index, recordings)
    plt.title('# of recordings for each command')
    plt.xlabel('Command name', fontsize=12)
    plt.ylabel('# of recordings', fontsize=12)
    plt.xticks(index, labels, fontsize=8, rotation=60)
    plt.show()

if __name__ == "__main__":
    number_of_data()