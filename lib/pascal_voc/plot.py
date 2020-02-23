import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_classes_count(dataset):
    sns.set_context("paper")
    groups = dataset.group_bounding_boxes_by('class_name')
    df = pd.DataFrame([(key, len(groups[key])) for key in groups.keys()], columns=['Class', 'Count'])
    df = df.sort_values(by=['Count'])
    plt.figure(figsize=(50, 10))
    chart = sns.barplot(x='Class', y='Count', data=df)
    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=25,
        horizontalalignment='right',
        fontsize='x-large'
    )
