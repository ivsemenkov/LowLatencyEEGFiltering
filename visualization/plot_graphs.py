import argparse
import os
from pathlib import Path
from typing import Optional
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from general_utilities.constants import RESULTS_ROOT
from visualization.create_plotting_data import get_experiment_parameters
from visualization.visualization_utils import get_algorithms_list, get_full_metric_name, get_margins, \
    get_simulation_name, get_simulation_names


def add_mean_labels(ax: plt.Axes, hue_order: list):
    """
    Add model indices to the graph

    Parameters
    ----------
    ax : Axes
                Matplotlib axis to add labels to
    hue_order : list
                A list of models in the order they appear in graph
    """
    lines = ax.get_lines()
    for idx, line in enumerate(lines):
        x, y = line.get_xdata(), line.get_ydata()
        text = ax.text(x.mean(), y.mean(), str((idx % len(hue_order)) + 1), ha='center', va='center',
                       fontweight='bold', color='white', fontsize=12)
        text.set_path_effects([
            path_effects.Stroke(linewidth=0.8, foreground=line.get_color()),
            path_effects.Normal(),
        ])


def add_p_values(ax: plt.Axes, p_values_data: pd.DataFrame, base_model: str, hue_order: list,
                 p_values_table: pd.DataFrame, metric_name: str, shift: float):
    """
    Adds p-values on graphs if base_model has better metrics than competitor

    Parameters
    ----------
    ax : Axes
                Matplotlib axis to add p-values to
    p_values_data : DataFrame
                A DataFrame with metrics data
    base_model : str
                A name of a base model to compare with
    hue_order : list
                A list of models in the order they appear in graph
    p_values_table : DataFrame
                A DataFrame with p-values data
    metric_name : str
                A metric which is on the ax
    shift : float
                Shift value for x axis on the plot
    """
    heights = {'Envelopes correlation': - 1,
               'Delay, ms': 1,
               'Circular standard deviation, degrees': 1}

    aggregs = {'Envelopes correlation': np.min,
               'Delay, ms': np.max,
               'Circular standard deviation, degrees': np.max}

    compars = {'Envelopes correlation': np.greater,
               'Delay, ms': np.less,
               'Circular standard deviation, degrees': np.less}

    lines = ax.get_lines()

    vals = {}
    base_x, base_y, base_order_num = None, None, None
    col = 'k'
    aggregation = aggregs[metric_name]
    comparison = compars[metric_name]
    critical_y = None
    maximal_value = None
    minimal_value = None

    for idx, model_name in enumerate(hue_order):
        line = lines[idx]
        model = model_name.split()
        order_num = int(model[0]) - 1
        name = ' '.join(model[1:])
        x, y = line.get_xdata(), line.get_ydata()
        x = x.mean() + shift
        y = aggregation(p_values_data.loc[p_values_data['Algorithm'] == model_name][metric_name])
        min_value = np.min(p_values_data.loc[p_values_data['Algorithm'] == model_name][metric_name].values)
        max_value = np.max(p_values_data.loc[p_values_data['Algorithm'] == model_name][metric_name].values)
        if critical_y is None:
            critical_y = y
        else:
            critical_y = aggregation([critical_y, y])
        if maximal_value is None:
            maximal_value = max_value
        else:
            maximal_value = np.max([maximal_value, max_value])
        if minimal_value is None:
            minimal_value = min_value
        else:
            minimal_value = np.min([minimal_value, min_value])
        if name == base_model:
            assert base_x is None and base_y is None and base_order_num is None
            base_x = x
            base_y = y
            base_order_num = order_num
        else:
            vals[model_name] = (x, y, order_num, name)

    prev_y = critical_y

    for competing_algo, (x, y, order_num, name) in vals.items():

        competing_algorithm_row = p_values_table.loc[p_values_table['Algorithm'] == name]
        assert len(competing_algorithm_row) == 1, competing_algorithm_row
        p_value = competing_algorithm_row['Mann-Whitney U two-sided rank test p-value'].values[0]
        base_mean = competing_algorithm_row['Base algorithm mean'].values[0]
        competing_algorithm_mean = competing_algorithm_row['Algorithm mean'].values[0]

        # If competitor better, do not draw p value to avoid confusion
        if comparison(competing_algorithm_mean, base_mean):
            continue

        if p_value < 0.0005:
            text = '***'
        elif p_value < 0.005:
            text = '**'
        elif p_value < 0.05:
            text = '*'
        else:
            text = 'n.s.'

        h = heights[metric_name] * (maximal_value - minimal_value) * 0.05
        y1 = prev_y + 0.5 * h
        y2 = y1 + h
        ax.plot([base_x, base_x, x, x], [y1, y2, y2, y1], lw=1.5, c=col)
        ax.text(np.min([base_x, x]) - 0.01, y2, text, ha='right', va='center', color=col)
        prev_y = y2


def find_indices_by_val(value: float, items_list: list):
    """
    Finds indices in items_list for which their respective items_list values surround value. Returns None if did not
    found

    Parameters
    ----------
    value : float
                A value which is inside indices
    items_list : list
                A list of values to find value in

    Returns
    -------
    low_idx : int
                Index of low boundary
    high_idx: int
                Index of higher boundary
    """
    for idx in range(len(items_list) - 1):
        if items_list[idx] <= value <= items_list[idx + 1]:
            low_idx = idx
            high_idx = idx + 1
            return low_idx, high_idx
    return None


def add_noise_labels(ax: plt.Axes, hue_order: list):
    """
    Add model indices to the noise robustness graph

    Parameters
    ----------
    ax : Axes
                Matplotlib axis to add labels to
    hue_order : list
                A list of models in the order they appear in graph
    """
    lines = ax.get_lines()
    noise_levels = [10, 15, 20, 25, 30, 50, 100]
    step = (noise_levels[-1] - noise_levels[0]) / len(hue_order)
    for idx, line in enumerate(lines):
        x, y = line.get_xdata(), line.get_ydata()
        if len(y) == 0:
            continue
        point_x = noise_levels[0] + idx * step
        low_idx, high_idx = find_indices_by_val(point_x, noise_levels)
        percentage_off = (point_x - noise_levels[low_idx]) / (noise_levels[high_idx] - noise_levels[low_idx])
        point_y = y[low_idx] + percentage_off * (y[high_idx] - y[low_idx])
        text = ax.text(point_x, point_y, str((idx % len(hue_order)) + 1), ha='center',
                       va='center', fontweight='bold', color='white', fontsize=8)
        text.set_path_effects([
            path_effects.Stroke(linewidth=0.8, foreground=line.get_color()),
            path_effects.Normal(),
        ])


def plot_main_graphs(metrics_to_calculate: list[str], save_name: str, csv_path: str, sampling_rate: float,
                     dataset_names: list[str], model_type: str, experiment_name: str, p_values_tables: Optional[dict],
                     gt_lag: int = 25):
    """
    Plots graphs for base noise level. Note: together with the main graph with all metrics it will save subgraphs for
    every used metric and for the legend in separate files. This is useful as such subgraphs can be inserted separately
    into LaTeX file and can be assigned different labels to make navigation easier

    Parameters
    ----------
    metrics_to_calculate : list
                A list of metrics to compute tests for. Available metrics: 'correlation', 'delay' and 'circstd_degrees'
    save_name : str
                A name of the saved graph
    csv_path : str
                A path to the csv with metrics for base noise levels
    sampling_rate : float
                Sampling rate of a data
    dataset_names : list
                Names of datasets to put on graph
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks
    experiment_name : str
                Name of the experiment which will be used to create path to graphs. Expected: sines, filtered_pink,
                state_space or multiperson_real_data
    p_values_tables : dict or None
                A dict with keys dataset names and values as DataFrames with p-values
    gt_lag : int
                    A lag of ground truth relative to inputs which should be compensated (applicable for forecasting)
    """
    sines_white_name = get_simulation_name('sines_white')
    sines_pink_name = get_simulation_name('sines_pink')
    no_correlation_simulations = [sines_white_name, sines_pink_name]

    metrics = pd.read_csv(csv_path)

    algorithms_to_use = get_algorithms_list(model_type=model_type)
    metrics = metrics.loc[metrics['Algorithm'].isin(algorithms_to_use)]
    if model_type == 'forecasting':
        base_model = 'Temporal Convolutional Network'
        if 'Delay, ms' in metrics.columns:
            metrics.loc[metrics['Algorithm'].isin(algorithms_to_use), 'Delay, ms'] -= 100.
        line_y = - 1000. * gt_lag / sampling_rate
    elif model_type == 'filtering':
        base_model = 'TCN filtering'
        line_y = 0.
    else:
        raise ValueError('Only available model types are filtering and forecasting')

    metrics = metrics.loc[metrics['Simulation'].isin(dataset_names)]
    simulation_names = get_simulation_names()
    metrics = metrics.replace({'Simulation': simulation_names})

    if p_values_tables is not None:
        updated_p_values_tables = {}
        for key, table in p_values_tables.items():
            updated_p_values_tables[simulation_names[key]] = table.replace({'Simulation': simulation_names})
    else:
        updated_p_values_tables = None

    legends = []
    bboxes = {}
    hue_order = []
    rename_dict = {}
    for idx, name in enumerate(algorithms_to_use):
        new_name = f'{idx + 1} {name}'
        rename_dict[name] = new_name
        hue_order.append(new_name)
    metrics = metrics.replace({'Algorithm': rename_dict})
    dx, dy, layout = get_margins(mode='main')
    if len(metrics.loc[~metrics['Simulation'].isin(no_correlation_simulations)]) == 0:
        n_rows = 1
        metrics_to_calculate = ['circstd_degrees']
        graphing_mode = 'sines'
    else:
        n_rows = 2
        graphing_mode = 'all_metrics'
    fig, axes = plt.subplots(n_rows, 2)
    if n_rows == 1:
        axes = np.array([axes])
    fig.set_size_inches((2 * dx, n_rows * dy), forward=False)
    for idx, metric_id in enumerate(metrics_to_calculate):
        full_metric_name = get_full_metric_name(metric_id=metric_id, visualization_mode=False)
        idx1, idx2 = divmod(idx, 2)
        df = metrics[['Algorithm', 'Simulation', full_metric_name]]
        if metric_id != 'circstd_degrees':
            df = df.loc[~df['Simulation'].isin(no_correlation_simulations)]
        bp = sns.boxenplot(df, x='Simulation', y=full_metric_name,
                           hue='Algorithm',
                           hue_order=hue_order,
                           linewidth=0, ax=axes[idx1, idx2],
                           saturation=1.,
                           flier_kws={'s': 0.1}
                           )
        add_mean_labels(ax=bp, hue_order=hue_order)

        if metric_id == 'correlation':
            # Taking 2 digits only for visualization
            axes[idx1, idx2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if updated_p_values_tables is not None:
            tick_labels = [t.get_text() for t in bp.get_xticklabels()]
            for tick, simulation_name in enumerate(tick_labels):
                if simulation_name in no_correlation_simulations:
                    # Errors for sines are so small that statistics calculations overflow. Not drawing p-values.
                    continue
                p_values_table = updated_p_values_tables[simulation_name]
                p_values_table = p_values_table.loc[p_values_table['Metric'] == full_metric_name]
                p_values_data = df.loc[df['Simulation'] == simulation_name]
                add_p_values(ax=bp, base_model=base_model, hue_order=hue_order,
                             p_values_data=p_values_data,
                             p_values_table=p_values_table,
                             metric_name=full_metric_name,
                             shift=tick)

        handles, labels = axes[idx1, idx2].get_legend_handles_labels()
        axes[idx1, idx2].get_legend().remove()
        if labels not in legends:
            legends.append(labels)
        assert len(legends) == 1
        if metric_id == 'delay':
            axes[idx1, idx2].axhline(y=line_y, color='r', linestyle='--')
        bp.set(xlabel=None, ylabel=get_full_metric_name(metric_id=metric_id, visualization_mode=True))
        coords = layout[graphing_mode][metric_id]
        bbox = mtransforms.Bbox(coords)
        bboxes[metric_id] = bbox
    labels = legends[0]
    axes[n_rows - 1, 1].axis('off')
    axes[n_rows - 1, 1].legend(handles, labels, loc='center', prop={'size': 8})
    coords = layout[graphing_mode]['legend']
    bbox = mtransforms.Bbox(coords)
    bboxes['legend'] = bbox
    fig.tight_layout()

    save_path = os.path.join(RESULTS_ROOT, 'paper_graphs', model_type, experiment_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for name, bbox in bboxes.items():
        fig.savefig(os.path.join(save_path, f'{name}_{save_name}.pdf'), bbox_inches=bbox)
    plt.savefig(os.path.join(save_path, f'{save_name}.pdf'), bbox_inches='tight')


def plot_noise_graphs(metrics_to_calculate: list[str], csv_path: str, sampling_rate: float, model_type: str,
                      dataset_name: str, gt_lag: int = 25):
    """
    Plots graphs for varying noise levels. Note: together with the main graph with all metrics it will save subgraphs
    for every used metric and for the legend in separate files. This is useful as such graphs can be inserted
    separately into LaTeX file and can be assigned different labels to make navigation easier

    Parameters
    ----------
    metrics_to_calculate : list
                A list of metrics to compute tests for. Available metrics: 'correlation', 'delay' and 'circstd_degrees'
    csv_path : str
                A path to the csv with metrics for varying noise levels
    sampling_rate : float
                Sampling rate of a data
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks
    dataset_name : str
                Name of a dataset to put on graph
    gt_lag : int
                    A lag of ground truth relative to inputs which should be compensated (applicable for forecasting)
    """
    algorithms_to_use = get_algorithms_list(model_type=model_type)

    hue_order = []
    rename_dict = {}
    for idx, name in enumerate(algorithms_to_use):
        new_name = f'{idx + 1} {name}'
        rename_dict[name] = new_name
        hue_order.append(new_name)

    fig, axes = plt.subplot_mosaic([metrics_to_calculate, len(metrics_to_calculate) * ['legend']],
                                   constrained_layout=True)
    bboxes = {}
    legends = []
    dx, dy, layout = get_margins(mode='noise')
    fig.set_size_inches((dx, 2 * dy), forward=False)

    metrics = pd.read_csv(csv_path)
    metrics = metrics.loc[metrics['Algorithm'].isin(algorithms_to_use)]
    metrics = metrics.loc[metrics['Simulation'] == dataset_name]

    if model_type == 'forecasting':
        metrics.loc[metrics['Algorithm'].isin(algorithms_to_use), 'Delay, ms'] -= 100.
        line_y = - 1000. * gt_lag / sampling_rate
    elif model_type == 'filtering':
        line_y = 0.
    else:
        raise ValueError('Only available model types are filtering and forecasting')

    metrics = metrics.replace({'Algorithm': rename_dict})

    for metric_idx, metric_id in enumerate(metrics_to_calculate):
        full_metric_name = get_full_metric_name(metric_id=metric_id, visualization_mode=False)
        df = metrics[['Noise level', 'Algorithm', full_metric_name]]
        lineplot = sns.lineplot(df, x='Noise level', y=full_metric_name, hue='Algorithm', errorbar='ci',
                                hue_order=hue_order, ax=axes[metric_id])
        add_noise_labels(ax=lineplot, hue_order=hue_order)
        handles, labels = axes[metric_id].get_legend_handles_labels()
        axes[metric_id].get_legend().remove()
        if labels not in legends:
            legends.append(labels)
        assert len(legends) == 1
        if metric_id == 'delay':
            axes[metric_id].axhline(y=line_y, color='r', linestyle='--')
        x_ticks = np.arange(10, 101, 10)
        axes[metric_id].set_xticks(x_ticks, labels=x_ticks, rotation=90)
        axes[metric_id].grid()
    bbox = mtransforms.Bbox(layout['all_metrics'])
    bboxes['graph'] = bbox
    axes[metrics_to_calculate[1]].set_title(model_type.title())

    labels = legends[0]
    for idx in range(len(metrics_to_calculate)):
        axes['legend'].axis('off')
    leg = axes['legend'].legend(handles, labels, loc='center', prop={'size': 8})
    for legobj in leg.legend_handles:
        legobj.set_linewidth(5.0)
    bbox = mtransforms.Bbox(layout['legend'])
    bboxes['legend'] = bbox
    save_path = os.path.join(RESULTS_ROOT, 'paper_graphs', model_type, 'noise')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_name = f'noise_{dataset_name}'
    for name, bbox in bboxes.items():
        fig.savefig(os.path.join(save_path, f'{name}_{save_name}.pdf'), bbox_inches=bbox)
    plt.savefig(os.path.join(save_path, f'{save_name}.pdf'))


def load_p_value_tables(p_value_filepaths: dict):
    """
    Loads and puts in a dict with respective simulations p-values info

    Parameters
    ----------
    p_value_filepaths : dict
                Dictionary with dataset_name as a key and path to respective p-value csv as a value

    Returns
    -------
    tables : dict
                Dictionary with dataset_name as a key and respective p-value DataFrame as a value
    """
    tables = {}

    for simulation_name, filepath in p_value_filepaths.items():
        df = pd.read_csv(os.path.join(RESULTS_ROOT, 'visualization_csv_files', filepath))
        df = df.loc[df['Simulation'] == simulation_name]
        assert len(np.unique(df['Simulation'])) == 1
        tables[simulation_name] = df

    return tables


def plot_all_graphs(visualization_csv_root: str, model_type: str):
    """
    Plot all graphs for multiple experiment names for the model type

    Parameters
    ----------
    visualization_csv_root : str
                A path to the directory with supplementary csv files. Example:
                path-to-project-root/results/visualization_csv_files
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration models and 'forecasting' for darts
                forecasting models
    """
    assert model_type in ['forecasting', 'filtering']

    gt_lag = 25
    sampling_rate = 250
    experiment_names = ['sines', 'filtered_pink', 'state_space']

    if model_type == 'filtering':
        experiment_names.append('multiperson_real_data')

    for experiment_name in experiment_names:
        dataset_names, metrics_to_calculate, save_name, stats_save_name, noise_dataset_name, \
            noise_save_name = get_experiment_parameters(experiment_name=experiment_name, model_type=model_type,
                                                        return_paths=False)

        main_csv_path = os.path.join(visualization_csv_root, f'{save_name}.csv')

        if experiment_name == 'sines':
            p_values_tables = None
        else:
            stats_path = os.path.join(visualization_csv_root, f'{stats_save_name}.csv')
            p_value_filepaths = {}
            for dataset_name in dataset_names:
                p_value_filepaths[dataset_name] = stats_path
            p_values_tables = load_p_value_tables(p_value_filepaths=p_value_filepaths)

        plot_main_graphs(metrics_to_calculate=metrics_to_calculate, save_name=save_name, csv_path=main_csv_path,
                         sampling_rate=sampling_rate, dataset_names=dataset_names, model_type=model_type,
                         experiment_name=experiment_name, p_values_tables=p_values_tables, gt_lag=gt_lag)

        if noise_save_name is not None:
            noise_csv_path = os.path.join(visualization_csv_root, f'{noise_save_name}.csv')
            plot_noise_graphs(metrics_to_calculate=metrics_to_calculate, csv_path=noise_csv_path,
                              sampling_rate=sampling_rate, model_type=model_type,
                              dataset_name=noise_dataset_name, gt_lag=gt_lag)


if __name__ == '__main__':
    sns.set_theme(style='white', font_scale=0.8)
    rc('font', **{'family': 'DeJavu Serif', 'serif': ['Computer Modern']})
    parser = argparse.ArgumentParser(description='Plots graphs from supplementary csv files.')
    parser.add_argument('-p', '--path', type=str, help='Path to the folder with supplementary csv files needed for '
                                                       'plotting. Typically, this will be '
                                                       'path-to-project-root/results/visualization_csv_files and csv '
                                                       'are created with create_plotting_data.py',
                        default=os.path.join(RESULTS_ROOT, 'visualization_csv_files'),
                        required=False)
    parser.add_argument('-m', '--model_type', type=str,
                        choices=['forecasting', 'filtering'],
                        help='Name of the model type for which the txt file is created. '
                             'Note: you should pass separately forecasting and filtering models in different txt files',
                        required=True)
    args = parser.parse_args()
    plot_all_graphs(visualization_csv_root=args.path, model_type=args.model_type)
