python ./src/download.py
docker swarm init
docker stack deploy -c docker-compose.yml my_stack
