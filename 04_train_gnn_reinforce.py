import os
import sys
import argparse
import pathlib
import ecole
import numpy as np
import parameters

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

    if int(args.load_n) == -1:
        load_model = f'model/imitation/mixed/train_params.pkl'
    else:
        load_model = f'model/imitation/{args.load_n}n/train_params.pkl'

    ### HYPER PARAMETERS ###
    max_epochs = 100000
    lr = 1e-4
    entropy_bonus = 0.0
    seed = parameters.seed
    tsp_size = int(args.num)

    if tsp_size == -1:
        if int(args.load_n) == -1:
            running_dir = f"model/reinforce/mixed-mixed"
        elif int(args.load_n) == 0:
            running_dir = f"model/reinforce/{tsp_size}n"
        else:
            running_dir = f"model/reinforce/{args.load_n}n-mixed"
    else:
        if int(args.load_n) == -1:
            running_dir = f"model/reinforce/mixed-{tsp_size}n"
        elif int(args.load_n) == 0:
            running_dir = f"model/reinforce/{tsp_size}n"
        else:
            running_dir = f"model/reinforce/{args.load_n}n-{tsp_size}n"

    os.makedirs(running_dir, exist_ok=True)

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    env = ecole.environment.Branching(
        reward_function=-1.5 * ecole.reward.LpIterations()**2,
        observation_function=ecole.observation.NodeBipartite(),
    )

    import torch
    from utilities import log
    sys.path.insert(0, os.path.abspath(f'model'))
    from model import GNNPolicy

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    ### LOG ###
    import time
    timestampstr = time.strftime('%m%d-%H%M')
    logfile = os.path.join(running_dir, f'train_log_{timestampstr}.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"lr: {lr}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {seed}", logfile)

    train_env = ecole.environment.Branching(
        scip_params={
            'separating/maxrounds': 0,
            'presolving/maxrestarts': 0,
            'limits/time': 3600,
            'timing/clocktype': 1,
            'branching/vanillafullstrong/idempotent': True
        },
        observation_function=ecole.observation.NodeBipartite(),
        reward_function=-ecole.reward.LpIterations(),
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "lpiters": ecole.reward.LpIterations().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
        },
    )

    valid_env = ecole.environment.Configuring(
        observation_function=None,
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
        },
        scip_params={
            "separating/maxrounds": 0,
            "presolving/maxrestarts": 0,
            "limits/time": 3600,
        },
    )

    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(load_model, map_location=device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    train_instances = [
        str(file)
        for file in (pathlib.Path(f'data/tsp{tsp_size}/instances') / f'train_{tsp_size}n').glob('instance_*.lp')
    ]
    valid_instances = [
        str(file)
        for file in (pathlib.Path(f'data/tsp{tsp_size}/instances') / f'valid_{tsp_size}n').glob('instance_*.lp')
    ]

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        instance = rng.choice(train_instances, size=1, replace=True)[0]

        # Run the GNN brancher
        observation, action_set, loss, done, info = train_env.reset(instance)
        while not done:
            with torch.no_grad():
                observation = (
                    torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                    torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                    torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                    torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
                )
                logits = policy(*observation)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                observation, action_set, reward, done, info = train_env.step(action)
                loss += reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log(
            f'TRAIN LOSS: {-loss:0.3f} ' + f"LP Iter: {info['lpiters']} " + f'time: {info["time"]:0.3f} ' +
            f'nb_nodes: {info["nb_nodes"]}', logfile)

        # TEST
        valid_env.reset(instance)
        _, _, _, reward, valid_info = valid_env.step({})

        log(
            f'VALID LOSS: {reward:0.3f} ' + f'LP Iter: {valid_info["lpiters"]} ' + f'time: {valid_info["time"]:0.3f} ' +
            f'nb_nodes: {valid_info["nb_nodes"]}', logfile)

    torch.save(policy.state_dict(), pathlib.Path(running_dir) / f'train_params_{timestampstr}.pkl')