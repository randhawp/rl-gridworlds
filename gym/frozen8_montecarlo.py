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


env = gym.make("FrozenLake-v0", desc=custom_map,is_slippery=False)
env.reset()
env.render()

np.set_printoptions(formatter={'float': '{: 0.9f}'.format})

#define the arrays to hold the state value and action values
P=np.zeros(16)  # holds the current policy
G=np.zeros(16) # holds state value temporarily until sweep is finished
episodecount=0
rewardstate=0

'''
first visit MonteCarlo only one value per state is stored during an episode
every visit MonteCarlo all visited states (duplicates or more) are retained
in the episode
'''
done=False
episodepath=[]
episodelen=0
visitedstate=np.array([])
lastval=0
prevstate=-1
while episodecount < 200: # number of episoded to play
    env.reset() # before each episode get back to starting state
    while 1:
        random_action = env.action_space.sample()
        state, reward, done, info = env.step(random_action)

        if prevstate==state:
            continue
        episodepath.append([state,reward])
        episodelen=episodelen+1
        prevstate = state
        if done and reward>0: #reached a reward state
            break
    #print("last episode len is ",episodelen)

    #calculate the value of each visited state from the goal to start
    episodepath_reversed=np.flip(episodepath)
    g=0

    for reward,state in episodepath_reversed:
        print(state,reward)
        state=int(state)
        G[state] = reward + 0.2 * g
        g = G[state]


    print("--------------------",episodecount,G[0])
    if(G[0] > lastval): # if staret state has the highest value
            P=np.copy(G) #then keep these state values for policy
            lastval = G[0]

    episodepath=[]
    teststate=[]
    visitedstate=[]
    episodecount+=1

print(P.reshape(4,4))
policy = evaluate_policy(env,P)
#printing policy array in sections to reshape it
#printing policy array in sections to reshape it
print(policy[0:4])
print(policy[4:8])
print(policy[8:12])
print(policy[12:16])
