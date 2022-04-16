import os
import sys
import argparse
import pathlib
import ecole
import numpy as np
import parameters


def pretrain(policy, pretrain_loader):
    policy.pre_train_init()
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr,
                                    batch.variable_features):
                break

        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(policy, data_loader, top_k=[1, 3, 5, 10], optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    mean_entropy = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
            entropy = (-F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = cross_entropy_loss - entropy_bonus * entropy

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            kacc = []
            for k in top_k:
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue
                pred_top_k = logits.topk(k).indices
                pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
                kacc.append(accuracy)
            kacc = np.asarray(kacc)
            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_entropy += entropy.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    mean_entropy /= n_samples_processed
    return mean_loss, mean_kacc, mean_entropy


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
        help='load model from file (specify by number of tsp size trained on). -1 if mixed trained model',
        type=int,
        default=15,
    )

    args = parser.parse_args()

    if int(args.load_n) == -1:
        load_model = f'model/imitation/mixed/train_params.pkl'
    else:
        load_model = f'model/imitation/{args.load_n}n/train_params.pkl'

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 1e-3
    entropy_bonus = 0.0
    top_k = [1, 3, 5, 10]
    seed = parameters.seed
    tsp_size = int(args.num)

    if tsp_size == -1:
        if int(args.load_n) == -1:
            running_dir = f"model/reinforce/mixed-mixed"
        else:
            running_dir = f"model/reinforce/{args.load_n}-mixed"
    else:
        if int(args.load_n) == -1:
            running_dir = f"model/reinforce/mixed-{tsp_size}n"
        else:
            running_dir = f"model/reinforce/{args.load_n}-{tsp_size}n"

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

    instances = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=200)

    for _ in range(10):
        observation, action_set, reward_offset, done, info = env.reset(next(instances))
        while not done:
            observation, action_set, reward, done, info = env.step(action_set[0])
