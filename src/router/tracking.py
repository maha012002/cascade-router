"""MLflow setup: local SQLite tracking DB + local artifact folder (no server needed)."""

import mlflow

from router.config import PROJECT_ROOT


def init_mlflow(tracking_uri: str, artifact_dir: str, experiment: str) -> None:
    """Point MLflow at the project-local DB and create the experiment if needed.

    MLflow >= 3.x refuses the plain-folder './mlruns' store by default, so we use SQLite.
    """
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = f"sqlite:///{(PROJECT_ROOT / tracking_uri.removeprefix('sqlite:///'))}"
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifact_root = (PROJECT_ROOT / artifact_dir).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(experiment)
