import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--tau',  default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=50000, type=int)  # replay buffer size
parser.add_argument('--num_iteration', default=100000, type=int)  # num of  games
parser.add_argument('--batch_size', default=256, type=int)  # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.1, type=float)
parser.add_argument('--noise_clip', default=0.2, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()

model_config = {
    'replay_size': 1048576*2,
}

train_config = {
    'batch_size': 256,
    'max_train_steps': 2e6,
    'learning_rate': 3e-4,
    'update_interval': 16,
    'gamma': 0.99,
    'tau': 0.005,
    'n_episodes': 10000,
    'max_trajectory_length': 10000,
    'exploration_noise': 0.1,
    'random_location_pro': 0.2,
    'seed': 1,
}

reward_config = {
    'arrival_reward': 1000.,
    'not_arrival_reward': -5000.,
    'target_dis_reward_scale': 1.,
    'bound_penalty': -100.
}

env_config = {
    'env_mode': 2,
}

map_scale = {
    'x_max': 600,
    'x_min': -100,
    'y_max': 3100,
    'y_min': 2500,
    'z_max': 300,
    'z_min': 30,
}
