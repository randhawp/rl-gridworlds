import gym
import numpy as np
'''
Cannonical example of finding the value of each state using policy iteration
and then using the state values found do a policy evaluation to find a policy.
This example is just to demonstrate polcy iteration and evaluation independently.
This is not how it is supposed to be done, and the next example will show how
policy evaluation can be used inside the iteration loop to converge faster.

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

'''
Modified argmax to return location of all highest values
'''
def findargmax( arr ):
    x = np.max(arr) #find highest 
    y = np.where(arr==x) #find index of 1 or n highest
    return  y[0].flatten().tolist(),len(y[0])


'''
Policy function gets the statevalue array (precalculated due to value iteration)
and then uses those values to return an array that has the state/action pairs
i.e direction to take from every state. 0/1/2/3 correspond to direction to take
A direction of -1 (no direction) is added for the terminal states
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
'''
def evaluate_policy(env,statevalue):
  i=0
  neighbourvalues=np.zeros(4)
  policy=[]
  while i < env.observation_space.n:
    j=0
    while j< env.action_space.n:
      nextstate = env.P[i][j][0][1]
      neighbourvalues[j]=statevalue[nextstate]
      j=j+1    
      
    directions,length = findargmax(neighbourvalues)
    if length==4:
      policy.append(-1)
    else:
      policy.append(directions)
    i=i+1
  return policy



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
Note 2 terminal states in this case.https://www.geeksforgeeks.org/numpy-argmax-python/

At the end of a sweep state update the utility values with the new ones calcuated.

Continue till the time convergence is achieved.

'''
i=j=0

v=np.zeros(16)  # holds the actual values for 4x4 states
vtemp=np.zeros(16) # holds values temporarily until a sweep of 4x4 states is finished
actionvalue=np.zeros(4) # array to store the value for a state due to actions in that state
convergencelimit = 0.0001
converged = False
reward = -1 # override the environment and change reward to -1 for each step
while not converged:
  i=0
  while i < env.observation_space.n: #sweep across the state space
    j=0
    while j< env.action_space.n:
      
      nextstate = env.P[i][j][0][1] #next state
      done = env.P[i][j][0][3] #done
      if done:
        reward=0   # will use our own rewards, 0 for done and -1 for every step
      else:
        reward=-1
      p=0.25 # override probability distribution and set every action to equal chance
      # 0.25 = 1 / (no of possible actions)
      
      actionvalue[j] = p * (reward + gamma*v[nextstate]) # value of this state for this action
      j=j+1

    vtemp[i] = np.sum(actionvalue).round(5) # value is the sum of all action value
    i=i+1
    
  #check if converged
  #calculate the diff between the two state spaces
  diff = v - vtemp
  diffav = abs(np.sum(diff))/(16)

  v = np.copy(vtemp) #sweep is finished, update the entire state space with new values
  if(diffav <= convergencelimit):
    break

print("The converged state values are as follows")
print(v.reshape(4,4))

print("------------------")
print("From the above state values we can find the policy")
policy = evaluate_policy(env,v)
#printing policy array in sections to reshape it
print(policy[0:4])
print(policy[4:8])
print(policy[8:12])
print(policy[12:16])