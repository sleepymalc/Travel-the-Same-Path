import os
import argparse
import csv
import numpy as np
import time
import parameters
import ecole
import pyscipopt

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
        help='tsp size train on. -1 if mixed training',
        type=int,
        default=15,
    )
    parser.add_argument(
        '-l',
        '--load_n',
        help=
        'load model from file (specify by number of tsp size trained on). -1 if mixed trained model, 0 to train from scratch',
        type=int,
        default=15,
    )

    args = parser.parse_args()
    tsp_size = int(args.num)

    instances = [{
        'type': f'tsp{tsp_size}',
        'path': f"data/tsp{tsp_size}/instances/test_{tsp_size}n/instance_{i+1}.lp"
    } for i in range(parameters.test_instance)]

    if tsp_size == 15:
        instances += [{
            'type': 'transfer-small',
            'path': f"data/tsp15/instances/test_7n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
        instances += [{
            'type': 'transfer-medium',
            'path': f"data/tsp15/instances/test_10n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
        instances += [{
            'type': 'transfer-big',
            'path': f"data/tsp15/instances/test_30n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
    elif tsp_size == 20:
        instances += [{
            'type': 'transfer-small',
            'path': f"data/tsp20/instances/test_10n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
        instances += [{
            'type': 'transfer-medium',
            'path': f"data/tsp20/instances/test_13n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
        instances += [{
            'type': 'transfer-big',
            'path': f"data/tsp20/instances/test_40n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
    elif tsp_size == 25:
        instances += [{
            'type': 'transfer-small',
            'path': f"data/tsp25/instances/test_12n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
        instances += [{
            'type': 'transfer-medium',
            'path': f"data/tsp25/instances/test_16n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]
        instances += [{
            'type': 'transfer-big',
            'path': f"data/tsp25/instances/test_50n/instance_{i+1}.lp"
        } for i in range(parameters.transfer_instance)]

    ### HYPER PARAMETERS ###
    seed = parameters.seed
    internal_branchers = ['relpscost']
    gnn_models = ['imitation']  # Can be supervised
    time_limit = 3600
    branching_policies = []

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

    print(f"tsp size: {args.load_n}-{tsp_size}")
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
    from model import GNNPolicy

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    loaded_calls = {}
    for policy in branching_policies:
        if policy['type'] == 'gnn':
            if policy['name'] not in loaded_models:
                ### MODEL LOADING ###
                model = GNNPolicy().to(device)
                if policy['name'] == 'imitation':
                    if int(args.load_n) == 0:
                        model.load_state_dict(
                            torch.load(f"model/imitation/{tsp_size}n/train_params.pkl", map_location=device))
                    elif int(args.load_n) == -1:
                        pass
                    else:
                        model.load_state_dict(
                            torch.load(f"model/reinforce/{args.load_n}n-{tsp_size}n /train_params.pkl",
                                       map_location=device))
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

    result_file = f"{args.load_n}-{tsp_size}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
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
