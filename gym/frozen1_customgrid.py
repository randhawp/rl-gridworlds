# frozen-lake-ex2.py
import gym
 
MAX_ITERATIONS = 4

custom_map = [
    'SFFHF',
    'HFHFF',
    'HFFFH',
    'HHHFH',
    'HFFFG'
]

env = gym.make("FrozenLake-v0", desc=custom_map,is_slippery=False)
env.reset()
env.render()
for i in range(MAX_ITERATIONS):
    random_action = env.action_space.sample()
    print("action is ",random_action)
    new_state, reward, done, info = env.step(random_action)
    print("new state is", new_state)
    env.render()
    if done:
        break