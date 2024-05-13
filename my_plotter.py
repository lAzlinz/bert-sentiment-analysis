import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def add_to_plot(values):
    num_of_plots = len(values.items())
    num_classes = len(next(iter(values.items()))[1])
    class_labels = [f'{i}' for i in range(num_classes)]
    bar_width = 0.5
    index = np.arange(num_classes)
    plt.figure(figsize=(16, 5))
    plt.subplots_adjust(left=0.062, right=0.98)

    for i, (key, value) in enumerate(values.items()):
        plt.subplot(1, num_of_plots, i+1)
        plt.bar(index, value, bar_width, color='yellow')
        plt.xlabel('class')
        plt.ylabel(key)
        plt.title(f'{key} by class')
        plt.xticks(index, class_labels)

    plt.savefig('./record/unbalanced_model_record.png')
    plt.show()

def add_confusion_matrix(values):
    plt.figure(figsize=(8, 6))
    sns.heatmap(values, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('predicted labels')
    plt.ylabel('true labels')
    plt.title('confusion matrix')
    plt.savefig('./record/unbalanced_model_record_confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    values = {
        'precision': [1,0,0.1],
        'recall': [0.88, 0.99, 1],
        'f1-score': [0.99, 0.76, 0.874545]
    }

    add_to_plot(values)


    values = [
        [10, 1, 2],
        [2, 12, 5],
        [5, 2, 15]
    ]

    add_confusion_matrix(values)
