import gym
import numpy as np
import operator
from IPython.display import clear_output

'''
The frozenlake env can be a good replacement to the grid world environments.
The frozenlake grid can have (S) start state (F) frozen or neutral state (H) hole
or -ive reward state and (G) goal state with +ive reward

The env can be made stochastic or deterministic using the is_slippery flag
If is_slippery is False then the transition probability will always be 1.0

The env can be navigated / sweeped using two ways
a) step function that actully moves the agent to the next cell
b) env.P function that can do a one step look ahead

State values are external to the enviroment and they have to be calculated.
The env only returns rewards and transition probabilities

Actions are
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
'''
#the is_slippery flag to make it stochastic is not
env = gym.make("FrozenLake-v0",is_slippery=False)
env.reset()                    
env.render()

#list how many actions
print("Action space: ", env.action_space.n)

#list how big is the state space
print("Observation space: ", env.observation_space.n)


'''
env.P[state][action] returns [(1.0, 0, 0.0, False)] 
as 
1.0 - probability of transitioning into next staticmethod
0   - next state 
1   - reward
False - done or not

Very useful to do a one step lookahead and check what rewards are possible
Lookahead when no action is actually taken
'''
print(env.P[0])

#or

state=3
action=2
print(env.P[state][action])

# taking an action
action=3
observation, reward, done, info = env.step(action)
print("onservation",observation) # returns the new state
print("reward",reward) #returns reward
print("status",done) # is the new state a terminal state
print("info",info) # return the transition probability

i=0
#looping through all states
while i < env.observation_space.n:
  print(env.P[i]) # print transition probability, next state,reward and status for each state
  i=i+1
#looping through all actions
j=0
while j < env.action_space.n:
  print(j) # print all actions
  j=j+1

# random action
'''
whilst action at each state can be decided by a policy
it can also be random
'''
random_action = env.action_space.sample()



