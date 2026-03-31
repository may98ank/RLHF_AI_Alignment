import os
import torch
from torch.distributions.categorical import Categorical
import gymnasium as gym
from matplotlib import pyplot as plt
from utils import mlp


def generate_rollout(policy, env, rendering=False):

    def get_action(policy, obs):
        with torch.no_grad():
            logits = policy(obs)
            return Categorical(logits=logits).sample().item()

    obs, _ = env.reset()
    done = False

    cum_ret = 0
    obs_traj = []
    while not done:

        if rendering:
            env.render()
        act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        cum_ret += rew
        obs_traj.append(obs)

    return obs_traj, cum_ret


def mean_return_for_checkpoint(checkpoint_path, env, num_rollouts):
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes = [32]
    policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    policy.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    policy.eval()
    total = 0.0
    for _ in range(num_rollouts):
        _, cum_ret = generate_rollout(policy, env, rendering=False)
        total += cum_ret
    return total / num_rollouts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='', help='single policy .params file')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='evaluate policy_checkpoint0..N-1; use with --plot for Part 3')
    parser.add_argument('--num_checkpoint_epochs', type=int, default=50)
    parser.add_argument('--plot', type=str, default='', help='save PNG when using --checkpoint_dir')
    parser.add_argument('--num_rollouts', type=int, default=1)

    args = parser.parse_args()

    env = gym.make(args.env_name)

    if args.checkpoint_dir:
        indices = []
        means = []
        for i in range(args.num_checkpoint_epochs):
            path = os.path.join(args.checkpoint_dir, 'policy_checkpoint%d.params' % i)
            if not os.path.isfile(path):
                print('missing checkpoint, stopping scan:', path)
                break
            m = mean_return_for_checkpoint(path, env, args.num_rollouts)
            print('checkpoint %d: average return (over %d rollouts) = %.3f' % (i, args.num_rollouts, m))
            indices.append(i)
            means.append(m)
        if args.plot and indices:
            plt.figure(figsize=(8, 5))
            plt.plot(indices, means, marker='o', markersize=3)
            plt.xlabel('Checkpoint (epoch index)')
            plt.ylabel('Average ground-truth return')
            plt.title('RL policy performance vs training checkpoint')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            plt.close()
            print('saved plot to', args.plot)
    else:
        if not args.checkpoint:
            raise SystemExit('Provide --checkpoint or --checkpoint_dir.')

        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n
        hidden_sizes = [32]
        policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
        policy.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        policy.eval()

        returns = 0
        for i in range(args.num_rollouts):
            _, cum_ret = generate_rollout(policy, env, rendering=args.render)
            print('cumulative return', cum_ret)
            returns += cum_ret
        print('average return', returns / args.num_rollouts)
