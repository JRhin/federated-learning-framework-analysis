import os

if __name__ == "__main__":
    import argparse

    description = """
    In this python file we perform a simulation by creating the passed number of clients that are going to subscribe to the master node.

    To check the available parameters just run `python main.py -h`. 
    """

    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c',
                        '--clients',
                        help='The number of maximum clients. Default 10.',
                        default=10,
                        type=int)
    args = parser.parse_args()
    
    os.system(f'docker service update --env-add CLIENTS={args.clients} my_stack_master')
    os.system(f'docker service scale my_stack_client={args.clients}')
