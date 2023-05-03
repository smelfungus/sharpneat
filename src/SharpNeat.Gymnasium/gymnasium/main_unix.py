import collections
import struct
import time
import traceback

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import ClipAction
from argparse import ArgumentParser
import numpy as np
import logging
import socket

logging.basicConfig(filename='debug.log', encoding='utf-8', level=logging.FATAL)
logging.debug("start")

parser = ArgumentParser()
parser.add_argument("-uuid", dest="uuid", default="test")
parser.add_argument("-render", dest="render", default="False")
parser.add_argument("-test", dest="test", default="False")
args = parser.parse_args()
render = args.render == "True"
test = args.test == "True"
socket_path = "/tmp/CoreFxPipe_sharpneat.gymnasium." + args.uuid + ".pipe"


# env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human" if render else None)
# env = gym.make("LunarLander-v2", render_mode="human" if render else None)
try:
    # env = gym.make("LunarLander-v2", enable_wind=True, render_mode="human" if render else None)
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human" if render else None)
    env = ClipAction(env)

    logging.debug("Environment created")
    logging.debug("Environment action size: %s", str(env.action_space.shape[0]))
    logging.debug("Environment action type: %s", str(env.action_space.dtype))
    logging.debug("Environment action type size: %s", str(env.action_space.dtype.itemsize))
    logging.debug("Environment action type char: %s", str(env.action_space.dtype.char))
except Exception as e:
    logging.error(e)


def run_episode():
    observation, info = env.reset()

    max_reward_history_len = 100
    total_reward = 0
    total_timesteps = 0
    latest_rewards = collections.deque(maxlen=max_reward_history_len)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    logging.debug("Sending initial observation")
    send_observation(sock, observation, 0, False)
    logging.debug("Initial observation sent")

    while True:
        logging.debug("Starting step")

        if not test:
            a = read_action(sock, env.action_space)
        else:
            a = env.action_space.sample()

        total_timesteps += 1

        observation, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # if reward != 0:
        #     print("reward %0.3f" % reward)

        total_reward += reward
        latest_rewards.append(float(reward))

        masked_done = done

        if total_timesteps >= max_reward_history_len:
            low_performing = True
            for historical_reward in latest_rewards:
                if historical_reward > 0:
                    low_performing = False
                    break
            if low_performing:
                masked_done = True

        if not test:
            send_observation(sock, observation, float(total_reward), masked_done)

        if render:
            env.render()
            time.sleep(0.01)

        if masked_done:
            logging.debug("Terminated")
            env.close()
            sock.close()
            break
            # print(reward)
            # input("Done")
    # print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


def send_observation(conn: socket, observation: np.array, reward: float, done: bool):
    message = bytes(observation.astype(float)) + bytes(np.array([reward]).astype(float)) + bytes(
            np.array([int(done)]))
    conn.sendall(message)
    # logging.debug("Observation sent: %s bytes", str(len(message)))


def read_action(conn: socket, space: spaces.Space):
    is_discrete = len(space.shape) == 0
    count = 1 if is_discrete else space.shape[0]
    type_char = 'i' if is_discrete else space.dtype.char
    item_size = 4 if is_discrete else space.dtype.itemsize
    action_struct = conn.recv(item_size * count)
    action_got = struct.unpack(count * type_char, action_struct)
    # logging.debug("Action read: %s", str(action_got))
    return action_got[0] if is_discrete else action_got


try:
    run_episode()
except Exception as e:
    logging.error(str(e))
    logging.error(traceback.format_exc())
