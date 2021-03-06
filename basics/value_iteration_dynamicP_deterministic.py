'''
VALUE ITERATION
- Synchronous Dynamic programming - that is sweep across all states
- Deterministic actions - i.e all possible actions have equal probability


Reward = -0.04 for non-termianl states
Gamma = 0.9
Two terminal states with value +1 and -1
The grid starts with all cells of the grid initialzed as zero and the terminal state remain zero

+--+--+--+----+
|  |  |  | +1 |
+--+--+--+----+
|  |  |  | -1 |
+--+--+--+----+
|  |  |  |    |
+--+--+--+----+

'''

import numpy as np 

# the board size can be changed
BOARD_ROW = 3
BOARD_COL = 4

START_ROW=2
START_COL=0
current_row=START_ROW
current_col=START_COL


# trow, tcol are x,y for the top right
trow = 0
tcol = BOARD_COL - 1

#init the states grid

#this is the store the state value (of every cell) of the individual sweep
states = np.zeros(BOARD_ROW*BOARD_COL).reshape(BOARD_ROW,BOARD_COL)

#this is to store the cumulative value after the sweep ends
statesnew = np.zeros(BOARD_ROW*BOARD_COL).reshape(BOARD_ROW,BOARD_COL)

states[trow,tcol]= 1 # terminal state value   
states[trow+1,tcol]= -1 # terminal state value

runningpolicy=np.chararray(BOARD_ROW * BOARD_COL) 
latestpolicy=np.chararray(BOARD_ROW * BOARD_COL) 
'''
Note on actions
The agent can move either up,down,left or right (4 possible actions)
From some cell on the border of the grid one or two actions may not be possible
as in the case of the top right and bottom left. All other edges have only 3 actions
Actions have equal probability of 0.25% each (0.25 * 4 = 100%)
The sum of probabilites of all actions has to be 100%

Each action has a reward of -1
'''
def getreward(row,col):
  reward = -0.04
  if row==trow and col == tcol:
      reward=1
  if row==trow+1 and col==tcol:
      reward=-1
  return reward

gamma=0.9
reward=0 
print(states)

t=0
convergencelimit = 0.0001
converged = False
c=4 #can start from any cell 0 is first (top left)
while not converged :
  row=0
  col=0
  
  temp=0
  while c < (BOARD_ROW * BOARD_COL  ): #loop through all the cells of grid 
    row=int(c/BOARD_COL)
    col=c-(row*BOARD_COL)
    
    #for each cell / state caluate the value for each action
    utilityvalue=np.zeros(4)

    if row==trow and col==tcol:
      statesnew[trow,tcol]= 1 # terminal state value   
      runningpolicy[c]='X'
      c=c+1
      continue
    if row==trow+1 and col==tcol:
      statesnew[trow+1,tcol]= -1 # terminal state value   
      runningpolicy[c]='X'
      c=c+1
      continue
      
    
    if row - 1 >=0: #top
      utilityvalue[0]=0.25 * (getreward(row-1,col) + gamma*(states[row-1,col]))
    else:
      utilityvalue[0]=0.25 *  (getreward(row,col) + gamma*(states[row,col]))
    
    if row + 1 < BOARD_ROW:#bottom
      utilityvalue[1]=0.25 * (getreward(row+1,col) + gamma*(states[row+1,col]))
    else:
      utilityvalue[1]=0.25 * (getreward(row,col) + gamma*(states[row,col]))
    
    if col - 1 >= 0:#left
      utilityvalue[2]=0.25* (getreward(row,col-1) + gamma*(states[row,col-1]))
    else:
      utilityvalue[2]=0.25 * (getreward(row,col) + gamma*(states[row,col]))
    
    if col + 1 < BOARD_COL: #right
      utilityvalue[3]=0.25* (getreward(row,col+1) + gamma*(states[row,col+1]))
    else:
      utilityvalue[3]=0.25 * (getreward(row,col) + gamma*(states[row,col]))


    reward = 0
    # add up the value for each of the possible actions in the current
    # cell and update statesnew (Note: do not update the current grid values
    # until full sweep of the entire state space is done)
    
    statesnew[row,col]= np.sum(utilityvalue)
    maxv = np.max(utilityvalue)
    result = np.where(utilityvalue == maxv)
    compass=''
    direction = (result[0][0])
    if direction == 0:
      compass = 'U'
    if direction == 1:
      compass = 'D'
    if direction == 2:
      compass = 'L'
    if direction == 3:
      compass = 'R'
       
    runningpolicy[c]=compass
    c=c+1
  
  
  #calculate the diff between the two state spaces
  diff = statesnew - states
  diffav = abs(np.sum(diff))/(BOARD_COL*BOARD_ROW)
  
  #after each pass update the state space with the new values  
  states = np.copy(statesnew)
  t=t+1  
  c=0
  #print((states))
  latestpolicy = runningpolicy

  #have the states converged (exit condition)
  if(diffav <= convergencelimit):
    break

print("Convergence reached after ", t , " steps")
print(states)
print(latestpolicy.reshape(BOARD_ROW,BOARD_COL).astype(str))
'''
[[-0.30734218 -0.17750818  0.31816232  1.        ]
 [-0.40129651 -0.44877186 -0.7068359  -1.        ]
 [-0.45373959 -0.53556325 -0.76908029 -1.23208465]]

Convergence reached after  27  steps

'''

