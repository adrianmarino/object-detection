import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def samples_count_by_class(dataset):
    groups = dataset.group_bounding_boxes_by('class_name')
    df = pd.DataFrame([(key, len(groups[key])) for key in groups.keys()], columns=['Class', 'Count'])
    df = df.sort_values(by=['Count'])
    return df


def missing_samples_count_by_class(dataset):
    groups = dataset.group_bounding_boxes_by('class_name')

    stats = [(key, len(groups[key])) for key in groups.keys()]
    stats.sort(key=lambda it: it[1], reverse=False)
    max_count = stats[-1][1]

    df = pd.DataFrame([(class_name, max_count - count, count) for class_name, count in stats],
                      columns=['Class', 'Missing', 'Count'])
    return df


def plot_samples_count_by_class(dataset):
    sns.set_context("paper")
    df = samples_count_by_class(dataset)
    plt.figure(figsize=(50, 10))
    chart = sns.barplot(x='Class', y='Count', data=df)
    chart.set_title('Samples count by class', fontsize='x-large')
    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=25,
        horizontalalignment='right',
        fontsize='x-large'
    )
    show_values_on_bars(chart.axes)


def plot_missing_samples_by_class(dataset):
    sns.set_context("paper")
    df = missing_samples_count_by_class(dataset)
    df = df[df['Missing'] > 0]

    if df.size == 0:
        print('Dataset balance!')
    else:
        plt.figure(figsize=(50, 10))
        chart = sns.barplot(x='Class', y='Missing', data=df)
        chart.set_title(f'Missing samples count by class to balance dataset to {df["Count"].max()} samples per class',
                        fontsize='x-large')
        chart.set_xticklabels(
            chart.get_xticklabels(),
            rotation=25,
            horizontalalignment='right',
            fontsize='x-large'
        )
        show_values_on_bars(chart.axes)


def show_values_on_bars(axs, h_v="v", space=0.1):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y + space, value, ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
