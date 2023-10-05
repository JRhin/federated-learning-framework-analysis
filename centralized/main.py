import os

clients = 10
os.system(f'docker service update --env-add CLIENTS={clients} my_stack_master')
os.system(f'docker service scale my_stack_client={clients}')
