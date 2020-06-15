import gym
import numpy as np
'''
MonteCarlo
'''
custom_map = [
    'SFFF',
    'FFFF',
    'FFFF',
    'FFFG'
]
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
TERMINAL STATE = -1
MOVE ANY DIR = 5
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
    if length==4 and neighbourvalues[0]==0:
      policy.append(-1)
    elif length==4:
      policy.append(5);
    else:
      policy.append(directions)
    i=i+1
  return policy


cstate=[] # current state value in a sweep
fstate=[] # final state value
#env = gym.make("FrozenLake-v0",desc=custom_map, is_slippery=False)
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
np.set_printoptions(formatter={'float': '{: 0.9f}'.format})
#hyperparameters
gamma=1.0 #discount factor
p=0.25 # deterministic probability distribution and set every action to equal chance
# change reward to -1 if there are many H.
reward=0 # lets not use the environment reward, our reward is -1 for every step
convergencelimit = 0.00001 # stop when state values differ less than this value

i=j=0

#define the arrays to hold the state value and action values
v=np.zeros(16)  # holds the cumulative values of states
vtemp=np.zeros(16) # holds state value temporarily until sweep is finished
actionvalue=np.zeros(4) # holds the actual individual value for each neghiboring state
converged = False
episodecount=0
rewardstate=0
goalstate_found=0
converged=False;
'''
first visit MonteCarlo only one value per state is stored during an episode
every visit MonteCarlo all visited states (duplicates or more) are retained
in the episode
'''
done=False
episodepath=[]
episodelen=0
visitedstate=np.array([])

while episodecount < 1: # number of episoded to play
    env.reset() # before each episode get back to starting state
    while 1:
        random_action = env.action_space.sample()
        nextstate, reward, done, info = env.step(random_action)
        teststate = np.where(visitedstate==nextstate)
        if len(teststate[0]>0):
            print(teststate[0],"found")
            continue

        visitedstate = np.append(visitedstate,nextstate)
        episodepath.append([nextstate,reward])
        episodelen=episodelen+1
        prevstate = nextstate
        if done and reward>0: #reached a reward state
            break
    #print("last episode len is ",episodelen)

    #calculate the value of each visited state from the goal to start
    episodepath_reversed=np.flip(episodepath)
    p=0

    for reward,state in episodepath_reversed:
        print(state,reward)

        state=int(state)
        vtemp[state] = reward + 0.2 * p
        p = vtemp[state]

    v=np.copy(vtemp)
    episodepath=[]
    episodecount+=1
    print("--------------------",episodecount)

print(v.reshape(4,4))
