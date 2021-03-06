import gym
import numpy as np
'''
Cannonical example of finding the value of each state using policy iteration.
A uniform random policy i.e every action has equal chance is applied.
There are 4 actions (up/down/left/right) and each have 25% chance
'''
# 4x4 grid - do not change size/shape as it is hardcoded in code below
# can change location of S
custom_map = [
    'GSFF',
    'FFFF',
    'FFFF',
    'FFFG'
]
# in above S can be anywhere there is a F. Only one S though



cstate=[] # current state value in a sweep
fstate=[] # final state value
env = gym.make("FrozenLake-v0", desc=custom_map,is_slippery=False)
env.reset()
env.render()

'''
Starting at a grid cell, sweep through the entire state space (i.e our policy)
For each calcualte the utility value v(s) till done (i.e reach terminal state)
Note 2 terminal states in this case.

At the end of a sweep state update the utility values with the new ones calcuated.

Continue till the time convergence is achieved.

'''
#hyperparameters
gamma=1.0  #discount factor
p=0.25 # deterministic probability distribution and set every action to equal chance
reward=-1 # lets not use the environment reward, our reward is -1 for every step
convergencelimit = 0.0001 # stop when state values differ less than this value

i=j=0

#define the arrays to hold the state value and action values
v=np.zeros(16)  # holds the cumulative values of states
vtemp=np.zeros(16) # holds state value temporarily until sweep is finished
actionvalue=np.zeros(4) # holds the actual individual value for each neghiboring state
converged = False

while not converged:
  i=0
  while i < env.observation_space.n: #sweep across the state space
    j=0
    while j< env.action_space.n:
      
      nextstate = env.P[i][j][0][1] #next state
      done = env.P[i][j][0][3] #done
      if done:
        actionvalue[j] = 0 # value of terminal state is zero
      else:
        actionvalue[j] = p * (reward + gamma*v[nextstate]) # value of this state for this action
      
      j=j+1

    vtemp[i] = np.sum(actionvalue) # value is the sum of all action value
    i=i+1
    
  #check if converged
  #calculate the diff between the two state spaces
  diff = v - vtemp
  diffav = abs(np.sum(diff))/(16)

  v = np.copy(vtemp) #sweep is finished, update the entire state space with new values
  if(diffav <= convergencelimit):
    break
v=np.round(v)
print(v.reshape(4,4))