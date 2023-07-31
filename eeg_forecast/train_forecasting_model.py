import logging
import os
import eco2ai
import hydra
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping
from eeg_forecast.available_models import AVAILABLE_DARTS_MODELS
from eeg_forecast.compatible_dataset import DartsCompatibleSyntheticDataset
from eeg_forecast.darts_test_metrics import get_test_metrics, forecast_with_darts
from general_utilities.constants import PROJECT_ROOT
from general_utilities.utils import fill_specific_params_dict, make_deterministic, save_json


@hydra.main(version_base=None, config_path='',
            config_name='')
def run_full_experiment(cfg: DictConfig):
    """
    Train one forecasting model on one dataset

    Parameters
    ----------
    cfg : DictConfig
                Configuration file for an experiment
    """
    dataset_path = os.path.join(PROJECT_ROOT, cfg['dataset_relative_path'])
    orig_wd = hydra.utils.get_original_cwd()
    current_wd = os.getcwd()
    rel_output_path = os.path.relpath(current_wd, orig_wd)
    output_root = os.path.join(PROJECT_ROOT, rel_output_path)
    start_from_checkpoint = cfg['start_from_checkpoint']
    ckpt_type = cfg['ckpt_type']
    dataloader_params = cfg['dataloader_params']
    model_params = cfg['model_params']
    model_name = cfg['model']
    opt_params = cfg['opt_params']
    sch_params = cfg['sch_params']
    early_stopping_params = cfg['early_stopping_params']
    additional_params = cfg['additional_params']
    dataset_common_params = cfg['dataset_common_params']
    train_dataset_specific_params = cfg['train_dataset_specific_params']
    val_dataset_specific_params = cfg['val_dataset_specific_params']
    gt_lag = dataset_common_params['gt_lag']
    batch_size = dataloader_params['batch_size']
    num_loader_workers = dataloader_params['num_loader_workers']
    fit_params = cfg['fit_params']
    pl_trainer_kwargs = OmegaConf.to_container(cfg['pl_trainer_kwargs'])

    my_stopper = EarlyStopping(**early_stopping_params)

    pl_trainer_kwargs['callbacks'] = [my_stopper]

    assert gt_lag == model_params['output_chunk_length'], 'Else, covariates size should be different'

    model = AVAILABLE_DARTS_MODELS[model_name](optimizer_cls=optim.AdamW,
                                               optimizer_kwargs=opt_params,
                                               lr_scheduler_cls=optim.lr_scheduler.ReduceLROnPlateau,
                                               lr_scheduler_kwargs=sch_params,
                                               model_name=model_name,
                                               pl_trainer_kwargs=pl_trainer_kwargs,
                                               loss_fn=nn.MSELoss(),
                                               batch_size=batch_size,
                                               work_dir=output_root,
                                               **additional_params,
                                               **model_params)

    if start_from_checkpoint:
        assert isinstance(start_from_checkpoint, str)
        ckpt_path = os.path.join(PROJECT_ROOT, start_from_checkpoint)
        if ckpt_type == 'best':
            best = True
        elif ckpt_type == 'last':
            best = False
        else:
            raise ValueError
        model = model.load_from_checkpoint(model_name=model_name, work_dir=ckpt_path, best=best)

    fill_specific_params_dict(common_params_dict=dataset_common_params,
                              specific_params_dict=train_dataset_specific_params)

    train_ds = DartsCompatibleSyntheticDataset(dataset_path=dataset_path, **train_dataset_specific_params)

    fill_specific_params_dict(common_params_dict=dataset_common_params,
                              specific_params_dict=val_dataset_specific_params)

    val_ds = DartsCompatibleSyntheticDataset(dataset_path=dataset_path, **val_dataset_specific_params)

    train_target_ts, train_covariates = train_ds.get_target_and_cov()
    val_target_ts, val_covariates = val_ds.get_target_and_cov()

    try:

        tracker = eco2ai.Tracker(
            project_name=model_name,
            experiment_description=f'training of {model_name}',
            file_name='train_emission.csv',
            alpha_2_code=cfg['alpha_2_country_code']
        )

        tracker.start()

        model.fit(train_target_ts, past_covariates=train_covariates,
                  val_series=val_target_ts, val_past_covariates=val_covariates,
                  num_loader_workers=num_loader_workers, **fit_params)

        tracker.stop()

    except KeyboardInterrupt:
        print('Fitting stopped, evaluating results')

    best_model = model.load_from_checkpoint(model_name=model_name, work_dir=output_root, best=True)

    tracker = eco2ai.Tracker(
        project_name=model_name,
        experiment_description=f'Validating {model_name}',
        file_name='validation_emission.csv',
        alpha_2_code=cfg['alpha_2_country_code']
    )

    tracker.start()

    prediction = forecast_with_darts(best_model, val_ds, gt_lag, num_loader_workers, batch_size)

    tracker.stop()

    test_metrics, errors_dict = get_test_metrics(gt_ts_list=val_target_ts,
                                                 predict_ts_list=prediction,
                                                 amount_of_steps=gt_lag,
                                                 input_chunk_length=cfg['model_params']['input_chunk_length'])

    print(errors_dict)

    best_model_path = model.trainer.checkpoint_callback.best_model_path
    test_metrics['best_model_path'] = best_model_path
    save_json(test_metrics, os.path.join(output_root, 'validation_metrics.json'))


if __name__ == '__main__':
    logging.getLogger('apscheduler.executors.default').propagate = False
    logging.getLogger('apscheduler.scheduler').propagate = False
    make_deterministic(seed=0)
    run_full_experiment()
