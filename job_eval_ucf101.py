import os
import urllib
import tarfile
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment, Dataset, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import PyTorchConfiguration, MpiConfiguration
from azureml.core.runconfig import DockerConfiguration
from azureml.tensorboard import Tensorboard


# get workspace
ws = Workspace.from_config()

# Connect to traning cluster
compute_name = 'cluster-a100-8gpus-80g'

compute_target = ComputeTarget(workspace=ws, name=compute_name)

# Connect to experiment
experiment_name = 'mmod-vicreg-finetune-protocol-ucf101'
exp = Experiment(workspace=ws, name=experiment_name)

# Connect to curated enviroment
enviroment_name = 'SSL-Pytorch-v1-12-6'
env = Environment.get(workspace=ws, name=enviroment_name)


# Connecto to dataset
dataset_name = 'ds_ucf101'
datastore_name = 'wsdatalake_ucf101'
datastore = ws.datastores[datastore_name]
dataset = Dataset.get_by_name(ws, name=dataset_name)

# Connect to Artifacts Datastore that contains pretrain model
artifacts_datastore = ws.datastores['workspaceartifactstore']
checkpoint = Dataset.File.from_files(path=(artifacts_datastore, 'ExperimentRun/dcid.mmod-vicregc-pretrain-ep200-4fps_1677253891_22c1bf03/logs/epoch=99-step=15700.ckpt'))

# get root of git repo
prefix = Path(__file__).parent

# training script
source_dir = str(prefix.joinpath('src'))
script_name = 'trainer.py'

# Number of GPUs per Node
devices = 8

# Number of nodes
num_nodes = 1

# Number of workers 
num_workers = 10

args = ['--data_path', dataset.as_named_input(dataset_name).as_download(), 
        '--checkpoint_path', checkpoint.as_named_input('checkpoint').as_download(),
        '--stage', 'eval',
        '--eval_protocol', 'finetune',
        '--eval_data_modality', 'video',
        '--eval_dataset', 'ucf101',
        '--eval_learning_rate', 0.2,
        '--eval_weight_decay', 0.0,
        '--eval_dropout_p', 0.8,
        '--eval_scheduler_type','step',
        '--max_epochs', 200,
        '--warmup_epochs', 0,
        '--accelerator', 'gpu',
        '--batch_size', 64,
        '--devices', devices,
        '--num_nodes', num_nodes,
        '--num_workers', num_workers]

# create distributed config
distr_config = MpiConfiguration(node_count=num_nodes, process_count_per_node=devices)

# Docker config
docker_config = DockerConfiguration(use_docker=True, shm_size='256g')

# create job config
src = ScriptRunConfig(
    source_directory=source_dir,
    script=script_name,
    arguments=args,
    compute_target=compute_name,
    environment=env,
    distributed_job_config=distr_config,
    docker_runtime_config=docker_config
)

# submit job
run = Experiment(ws, experiment_name).submit(src)   

