# Create a logger that will use mlflow
# This is heavily based on the PyTorch Lightning MLFlow logger
# https://github.com/PyTorchLightning/PyTorch-Lightning/blob/master/pytorch_lightning/loggers/mlflow.py

import tianshou
import numpy as np
from datetime import datetime
from pathlib import Path
import torch

from numbers import Number
from typing import Callable, Dict, Optional, Tuple, Union

from tianshou.data import Batch

from mlflow.tracking import MlflowClient
import mlflow.projects.utils as mlflow_utils
from mlflow.projects.utils import *

LOG_DATA_TYPE = Dict[str, Union[int, Number, np.number, np.ndarray]]
LOCAL_FILE_URI_PREFIX = "file:"

# Function to flatten dictionaries to get a better logging name space for mlflow
def process_nested_dict(d, delimiter="--", max_level=2):
    """max_level is the maximum number of recursive calls"""

    if dict not in [type(v) for v in d.values()]:
        return d
    elif max_level <= 0:
        return d
    else:
        out_d = {}
        for k, v in d.items():
            if type(v) == dict:
                for subk, subv in v.items():
                    if subk == "_target_":
                        out_d[k] = subv
                    else:
                        out_d[f"{k}{delimiter}{subk}"] = subv
            else:
                out_d[k] = v
        return process_nested_dict(out_d, delimiter, max_level - 1)


class TianshouMLFlowLogger(tianshou.utils.BaseLogger):
    def __init__(
        self,
        train_interval=1000,
        test_interval=1,
        update_interval=1000,
        experiment_name="Default",
        run_name=None,
        tracking_uri=None,
        tags=None,
        save_dir="./mlruns",
        prefix="",
        artifact_location=None,
        filename=None,
        info_logger=None,
        policy=None,
        model_checkpoints=False,
        cp_path="",
    ):
        super().__init__(train_interval, test_interval, update_interval)
        if not tracking_uri:
            tracking_uri = f"{LOCAL_FILE_URI_PREFIX}{save_dir}"

        self._experiment_name = experiment_name
        self._experiment_id = None
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._run_id = None
        self.tags = self._get_mlflow_tags(filename=filename, manual_tags=tags)
        self._prefix = prefix
        self._artifact_location = artifact_location
        self.info_logger = info_logger

        self.policy = policy
        self.model_checkpoints = model_checkpoints
        self.best_test_reward = None

        if self.model_checkpoints:
            if cp_path is None:
                now_day = datetime.strftime(datetime.now(), "%Y-%m-%d")
                now_time = datetime.strftime(datetime.now(), "%H-%M-%S")
                self.cp_path = Path(f"./model_checkpoints/{now_day}/{now_time}/")
            else:
                self.cp_path = Path(cp_path)
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self._mlflow_client = MlflowClient(tracking_uri)

    @property
    def experiment(self):
        """
        Actual MLflow object
        Example::
            self.logger.experiment.some_mlflow_function()
        """
        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None:
                self._experiment_id = expt.experiment_id
            else:
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name,
                    artifact_location=self._artifact_location,
                )

        if self._run_id is None:
            if self._run_name is not None:
                self.tags[MLFLOW_RUN_NAME] = self._run_name
            run = self._mlflow_client.create_run(
                experiment_id=self._experiment_id, tags=self.tags
            )
            self._run_id = run.info.run_id

        e = self._mlflow_client.get_experiment(self._experiment_id)
        return self._mlflow_client

    @property
    def run_id(self):
        """Create the experiment if it does not exist to get the run id.
        Returns:
            The run id.
        """
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self):
        """Create the experiment if it does not exist to get the experiment id.
        Returns:
            The experiment id.
        """
        _ = self.experiment
        return self._experiment_id

    def log_hyperparameters(self, params):
        params_to_log = process_nested_dict(params)
        for k, v in params_to_log.items():
            if len(str(v)) > 250:
                f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}", RuntimeWarning
                continue
            self.experiment.log_param(self.run_id, k, v)

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        """Specify how the writer is used to log data.

        :param str step_type: namespace which the data dict belongs to.
        :param int step: stands for the ordinate of the data dict.
        :param dict data: the data to write with format ``{key: value}``.
        """
        for k, v in data.items():
            self.experiment.log_metric(self._run_id, k, v, step)

    def close(self) -> None:
        """"""
        self.experiment.set_terminated(self._run_id)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.
        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        .. note::
            ``collect_result`` will be modified in-place with "rew", "rew_std", "len",
            and "len_std" keys.
        """
        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                "test/env_step": step,
                "test/reward": rew,
                "test/length": len_,
                "test/reward_std": rew_std,
                "test/length_std": len_std,
            }

            # Supplement the data to be logged with stuff from info
            if self.info_logger:
                info_to_log = self.info_logger.report_for_logging()
                for k, v in info_to_log.items():
                    log_data[k] = v

            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step

            if self.model_checkpoints:
                if self.best_test_reward is None or rew > self.best_test_reward:
                    self.best_test_reward = rew
                    torch.save(
                        self.policy.state_dict(),
                        self.cp_path.joinpath(
                            f"step_{step}_rew_{self.best_test_reward}.dat"
                        ),
                    )

    @staticmethod
    def _get_mlflow_tags(filename=None, manual_tags=None):
        # Can specify filename as string
        # for example for Jupyter where os.path.basename(__file__)
        # does not work

        # Use specified filename if provided
        # Otherwise resolve automatically
        if filename:
            source_name = filename
        else:
            source_name = resolve_tags()["mlflow.source.name"]

        # Use specified working directory if provided

        work_dir = os.getcwd()

        source_version = mlflow_utils._get_git_commit(work_dir)
        tags = {
            MLFLOW_USER: mlflow_utils._get_user(),
            MLFLOW_SOURCE_NAME: source_name,
        }
        if source_version is not None:
            tags[MLFLOW_GIT_COMMIT] = source_version

        repo_url = mlflow_utils._get_git_repo_url(work_dir)
        if repo_url is not None:
            tags[MLFLOW_GIT_REPO_URL] = repo_url
            tags[LEGACY_MLFLOW_GIT_REPO_URL] = repo_url

        if manual_tags:
            for k, v in manual_tags.items():
                tags[k] = v

        return tags


# Creat InfoLogger object to record extra stuff about the run


class InfoLogger:
    def __init__(self):
        # Track mean values from episodes, aggregated when logged
        self._reset_after_log()

    def _reset_after_log(self):
        self.expiries = np.array([])
        self.backorders = np.array([])
        self.units_in_stock = np.array([])

    def preprocess_fn(self, **kwargs):
        # If it's a normal step
        if "rew" in kwargs:
            self.expiries = np.hstack([self.expiries, kwargs["info"]["daily_expiries"]])
            self.backorders = np.hstack(
                [self.backorders, kwargs["info"]["daily_backorders"]]
            )
            self.units_in_stock = np.hstack(
                [self.units_in_stock, kwargs["info"]["units_in_stock"]]
            )
            return Batch()
        else:
            return Batch()

    def report_for_logging(self):
        to_log = {
            "test/mean_daily_expiries": np.mean(self.expiries),
            "test/mean_daily_backorders": np.mean(self.backorders),
            "test/mean_daily_units_in_stock": np.mean(self.units_in_stock),
        }
        self._reset_after_log()

        return to_log
