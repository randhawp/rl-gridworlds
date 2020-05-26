import gym
import numpy as np
import operator
from IPython.display import clear_output

def create_random_policy(env):
     policy = {}
     for key in range(0, env.observation_space.n):
          current_end = 0
          p = {}
          for action in range(0, env.action_space.n):
               p[action] = 1 / env.action_space.n
          policy[key] = p
     return policy


env = gym.make("FrozenLake-v0")
env.reset()                    
env.render()

policy={}
policy = create_random_policy(env)
<<<<<<< HEAD
#print(policy)

print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)
=======
print(policy)
>>>>>>> ac67edece9762d1279a8706983fa32460be6b0bf
