# How to run

Is high recommended to firstly create a virtual enviroment ([here](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/) a guide).

Then the launching procedure changes respect to the flavor of Federated Learning.

## Centralized Federated Learning (CFL)

To launch a CFL simulation simply run the following commands:

```bash
.\run.sh
python main.py
```

This will run a simulation with 10 worker nodes (10 hospitals). If you want to run a personalized simulation you can use the `-c` flag to set the number of clients.

```bash
# Example
.\run.sh
python main.py -c 20 
```

> This will run a simulation with 20 worker nodes (20 hospitals).

## Multi-Mastered Federated Learning (MMFL)

To launch a MMFL simulation you have to have multiple systems (for masters and workers) to create different docker swarm.

In each master system you have to run:

```bash
.\run.sh
```

The copy the command with the token needed to add each worker to the swarm.

To add a worker system to a node just paste the copied command in a terminal a run it. The command has the following structure:

```bash
docker swarm join --token <TOKEN> <ADDRESS>:2377
```
Then from each master system, run:

```bash
python master.py
```

> As for the CLF you can change the number of workes (hospitals) by setting the `CLIENTS` variable using the `-c` flag.

## Closing a simulation

Remember to exit the Docker Swarm by running in each (if more than one) terminal:

```bash
docker swarm leave --force
```

> Note: `--force` is needed only in the terminal where the swarm was launched.