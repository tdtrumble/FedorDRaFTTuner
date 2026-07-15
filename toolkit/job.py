from typing import Union, OrderedDict

from toolkit.config import get_config


def get_job(
        config_path: Union[str, dict, OrderedDict],
        name=None
):
    config = get_config(config_path, name)
    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job = config['job']
    if job == 'krea2_draft':
        from jobs import ExtensionJob
        return ExtensionJob(config)
    raise ValueError("This fork only accepts job: krea2_draft")


def run_job(
        config: Union[str, dict, OrderedDict],
        name=None
):
    job = get_job(config, name)
    job.run()
    job.cleanup()
