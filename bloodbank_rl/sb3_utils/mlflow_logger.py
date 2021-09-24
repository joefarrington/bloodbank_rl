from stable_baselines3.common.logger import *
import mlflow.projects.utils as mlflow_utils
from mlflow.projects.utils import *

# We should be able to enhance this to log the other types of things
# just need to look at the documentation
class MlflowOutputFormat(KVWriter):
    def __init__(self, mlflow_client, mlflow_run_id):
        """
        Mlflow subsitute for logger
        """
        self.mlflow_client = mlflow_client
        self.mlflow_run_id = mlflow_run_id

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "tensorboard" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    pass
                else:
                    self.mlflow_client.log_metric(self.mlflow_run_id, key, value, step)

            if isinstance(value, th.Tensor):
                pass

            if isinstance(value, Video):
                pass

            if isinstance(value, Figure):
                pass

            if isinstance(value, Image):
                pass

    def close(self) -> None:
        """"""
        self.mlflow_client.set_terminated(self.mlflow_run_id)


# This is a little bit hacked together at the moment
def get_mlflow_tags(uri, experiment_id, work_dir, filename):

    source_name = filename

    source_version = mlflow_utils._get_git_commit(work_dir)
    existing_run = fluent.active_run()
    if existing_run:
        parent_run_id = existing_run.info.run_id
    else:
        parent_run_id = None

    tags = {
        MLFLOW_USER: mlflow_utils._get_user(),
        MLFLOW_SOURCE_NAME: source_name,
    }
    if source_version is not None:
        tags[MLFLOW_GIT_COMMIT] = source_version
    if parent_run_id is not None:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id

    repo_url = mlflow_utils._get_git_repo_url(work_dir)
    if repo_url is not None:
        tags[MLFLOW_GIT_REPO_URL] = repo_url
        tags[LEGACY_MLFLOW_GIT_REPO_URL] = repo_url

    return tags