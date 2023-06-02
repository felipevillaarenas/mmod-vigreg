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
compute_name = 'cluster-v100-8gpus-32g' #'cluster-v100-4gpus-32g' 'cluster-a100-8gpus-80g'  'cluster-v100-8gpus-32g' 'cluster-a100-8gpus-40g'
compute_target = ComputeTarget(workspace=ws, name=compute_name)

# Connect to experiment
experiment_name = 'mmod-vicreg-eval-ucf101'
exp = Experiment(workspace=ws, name=experiment_name)

# Connect to curated enviroment
enviroment_name = 'SSL-Pytorch-v2'
env = Environment.get(workspace=ws, name=enviroment_name)

# Connecto to dataset
dataset_ucf101 = Dataset.get_by_name(ws, name='ucf_101')
dataset_byolweights = Dataset.get_by_name(ws, name='byolweights')

# Connect to Artifacts Datastore that contains pretrain model
artifacts_datastore = ws.datastores['workspaceartifactstore']
checkpoint = Dataset.File.from_files(path=(artifacts_datastore, 'ExperimentRun/dcid.mmod-vicreg-train-pretrainedbackbones_1685625805_e0efba92/outputs/model.pt'))

# get root of git repo
prefix = Path(__file__).parent

# training script
source_dir = str(prefix.joinpath('../src'))
script_name = 'trainer.py'

# Number of GPUs per Node
devices = 8

# Number of nodes
num_nodes = 1

# Number of workers 
num_workers = 5

args = ['--data_path', dataset_ucf101.as_named_input('ucf_101').as_download(), 
        '--checkpoint_path', checkpoint.as_named_input('checkpoint').as_download(),
        '--path_pretrained_backbone_weights',dataset_byolweights.as_named_input('byolweights').as_download(),
        '--stage', 'eval',
        '--eval_protocol', 'linear',
        '--eval_data_modality', 'video',
        '--eval_dataset', 'ucf101',
        '--eval_learning_rate', 0.05,
        '--eval_weight_decay', 0.1,
        '--eval_dropout_p', 0.0,
        '--max_epochs', 20,
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

