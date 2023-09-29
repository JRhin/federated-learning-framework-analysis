#!/bin/bash

echo -e 'Installing needed python requirements...'
pip install -q -r requirements.txt

echo -e '\n\nDownload and partitioning the data...'
python ./src/download.py

echo -e '\n\nInitializing a Swarm'
docker swarm init

echo -e '\n\nDeploying a stack of containers'
docker stack deploy -c docker-compose.yml my_stack

echo -e '\nDONE'
