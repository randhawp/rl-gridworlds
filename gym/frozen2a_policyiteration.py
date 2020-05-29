import gym
import numpy as np
'''
In froze2_policyiteration the result on convergence is not

+-----+-----+-----+-----+
|  0  | -14 | -20 | -22 |
+-----+-----+-----+-----+
| -14 | -18 | -20 | -20 |
+-----+-----+-----+-----+
| -20 | -20 | -18 | -14 |
+-----+-----+-----+-----+
| -22 | -20 | -14 |  0  |
+-----+-----+-----+-----+
as in Richard S. Sutton and Andrew G. Bartos bible on RL
but it is
 [[ 0. -13. -19. -21.]
 [-13. -17. -19. -19.]
 [-19. -19. -17. -13.]
 [-21. -19. -13.   0.]]

This is because the terminal states are included in the evaluation.
Different state values do not matter as long as they are all proportional
and the policy still converges

In this example we exclude the terminal states to arrive at the result in the book
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

gamma=1.0  #discount factor
reward=0

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
i=j=0

v=np.zeros(16)  # holds the actual values
vtemp=np.zeros(16) # holds values temporarily until sweep is finished
actionvalue=np.zeros(4) # array to store the value for a state due to actions in that state
convergencelimit = 0.0001
converged = False
reward = -1 # override the environment and change reward to -1 for each step
c=0
p=0.25 # override probability distribution and set every action to equal chance

while not converged:
  i=1
  while i < env.observation_space.n -1: #sweep across the state space
    j=0
    while j< env.action_space.n:
      
      nextstate = env.P[i][j][0][1] #next state
      done = env.P[i][j][0][3] #done
      
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