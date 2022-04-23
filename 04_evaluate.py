import os
import argparse
import csv
import numpy as np
import time
import parameters
import ecole
import pyscipopt
from concorde.tsp import TSPSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g',
        '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-n',
        '--num',
        help='tsp size test on.',
        type=int,
        default=15,
    )
    parser.add_argument(
        '-l',
        '--load',
        help='load model trained on the size specified.',
        type=int,
        default=15,
    )

    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    normal_size = 100

    seed = parameters.seed

    internal_branchers = ['relpscost']
    gnn_models = ['supervised']
    time_limit = parameters.time_limit
    branching_policies = []
    tsp_size = int(args.num)
    imitation_size = int(args.load)

    instances = [{
        'type': f'tsp{tsp_size}',
        'path': f"data/tsp{tsp_size}/instances/test/instance_{i+1}.lp"
    } for i in range(normal_size)]

    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        branching_policies.append({
            'type': 'internal',
            'name': brancher,
        })

    # GNN models
    for model in gnn_models:
        branching_policies.append({
            'type': 'gnn',
            'name': model,
        })

    print(f"tsp size tested on: {tsp_size}")
    print(f"tsp size trained on: {imitation_size}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    import torch
    import torch_geometric
    from model import GNNPolicy

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    loaded_calls = {}
    for policy in branching_policies:
        if policy['type'] == 'gnn':
            if policy['name'] not in loaded_models:
                ### MODEL LOADING ###
                model = GNNPolicy().to(device)
                if policy['name'] == 'supervised':
                    model.load_state_dict(
                        torch.load(f"model/imitation/{imitation_size}n/train_params.pkl", map_location=device))
                else:
                    raise Exception(f"Unrecognized GNN policy {policy['name']}")

            loaded_models[policy['name']] = model

            policy['model'] = loaded_models[policy['name']]

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'walltime',
        'proctime',
    ]
    os.makedirs('results', exist_ok=True)
    scip_parameters = {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        'limits/time': time_limit,
        'timing/clocktype': 1,
        'branching/vanillafullstrong/idempotent': True
    }

    result_file = f"{args.load}-on-{tsp_size}n_{time.strftime('%m%d-%H%M')}.csv"
    with open(f"results/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                if policy['type'] == 'internal':
                    # Run SCIP's default brancher
                    env = ecole.environment.Configuring(scip_params={
                        **scip_parameters, f"branching/{policy['name']}/priority": 9999999
                    })
                    env.seed(seed)

                    walltime = time.perf_counter()
                    proctime = time.process_time()

                    env.reset(instance['path'])
                    _, _, _, _, _ = env.step({})

                    walltime = time.perf_counter() - walltime
                    proctime = time.process_time() - proctime

                elif policy['type'] == 'gnn':
                    # Run the GNN policy
                    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                                      scip_params=scip_parameters)
                    env.seed(seed)
                    torch.manual_seed(seed)

                    walltime = time.perf_counter()
                    proctime = time.process_time()

                    observation, action_set, _, done, _ = env.reset(instance['path'])
                    while not done:
                        with torch.no_grad():
                            observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                                           torch.from_numpy(observation.edge_features.indices.astype(
                                               np.int64)).to(device),
                                           torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(
                                               -1, 1).to(device),
                                           torch.from_numpy(observation.column_features.astype(np.float32)).to(device))

                            logits = policy['model'](*observation)
                            action = action_set[logits[action_set.astype(np.int64)].argmax()]
                            observation, action_set, _, done, _ = env.step(action)

                    walltime = time.perf_counter() - walltime
                    proctime = time.process_time() - proctime

                scip_model = env.model.as_pyscipopt()
                stime = scip_model.getSolvingTime()
                nnodes = scip_model.getNNodes()
                nlps = scip_model.getNLPs()
                gap = scip_model.getGap()
                status = scip_model.getStatus()

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': seed,
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'walltime': walltime,
                    'proctime': proctime,
                })
                csvfile.flush()

                print(
                    f"  {policy['type']}:{policy['name']} {seed} - {nnodes} nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}"
                )

    # Evaluate Concorde result
    random = np.random
    random.seed(seed)

    n = normal_size
    filenames = [os.path.join(f'data/tsp{tsp_size}/instances/test', f'instance_{i+1}.lp') for i in range(n)]
    nums = [tsp_size] * n
    with open(f"results/concorde/{tsp_size}n_{time.strftime('%m%d-%H%M')}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['instance', 'walltime'])
        writer.writeheader()
        # actually generate the instances
        for filename, num in zip(filenames, nums):
            print(f'  evaluate {filename} by Concorde...')
            coord_x = random.rand(num) * 100
            coord_y = random.rand(num) * 100

            start_time = time.time()
            solver = TSPSolver.from_data(coord_x, coord_y, norm="GEO")
            solution = solver.solve()
            end_time = time.time() - start_time
            writer.writerow({
                'instance': filename,
                'walltime': end_time,
            })
            csvfile.flush()
