from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment, Dataset, ComputeTarget
from azureml.core.runconfig import MpiConfiguration, DockerConfiguration


# get workspace
ws = Workspace.from_config("config/workspace.json")

# Connect to traning cluster
compute_name = 'cluster-v100-8gpus-32g' #'cluster-v100-4gpus-32g' 'cluster-a100-8gpus-80g'  'cluster-v100-8gpus-32g' 'cluster-a100-8gpus-40g'
compute_target = ComputeTarget(workspace=ws, name=compute_name)

# Connect to experiment
experiment_name = 'mmod-vicreg-train-pretrainedbackbones'
exp = Experiment(workspace=ws, name=experiment_name)

# Connect to curated enviroment
enviroment_name = 'SSL-Pytorch-v2'
env = Environment.get(workspace=ws, name=enviroment_name)

# Connecto to dataset
dataset_kinetics400 = Dataset.get_by_name(ws, name='kinetics400')
dataset_byolweights = Dataset.get_by_name(ws, name='byolweights')

# training script
prefix = Path(__file__).parent
source_dir = str(prefix.joinpath('../src'))
script_name = 'trainer.py'

# Number of GPUs per Node, Nodes and Workers per GPU
devices = 8
num_nodes = 1
num_workers = 5

args = ['--data_path', dataset_kinetics400.as_named_input('kinetics400').as_download(),
        '--devices', devices,
        '--num_nodes', num_nodes,
        '--num_workers', num_workers,
        '--path_pretrained_backbone_weights',dataset_byolweights.as_named_input('byolweights').as_download(),
        ]

# Create distributed config
distr_config = MpiConfiguration(node_count=num_nodes, process_count_per_node=devices)

# Docker config
docker_config = DockerConfiguration(use_docker=True, shm_size='16g')

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
