'''
- Synchronous Dynamic programming - that is sweep across all states
- Deterministic actions - i.e all possible actions have equal probability

This example illustrates policy iteration with with the help of a [ n x n ] grid
The top left and bottom right are terminal states with value 0
Reward = -1
Gamma = 1
Two terminal states with value 0
The grid starts with all cells of the grid initialzed as zero and the terminal state remain zero
+---+--+--+---+
| 0 |  |  |   |
+---+--+--+---+
|   |  |  |   |
+---+--+--+---+
|   |  |  |   |
+---+--+--+---+
|   |  |  | 0 |
+---+--+--+---+

'''

import numpy as np 

# the board size can be changed
BOARD_ROW = 4
BOARD_COL = 4

START_ROW=0
START_COL=0
current_row=START_ROW
current_col=START_COL


# trow, tcol are x,y for the bottom right terminal states
trow = BOARD_ROW -1 
tcol = BOARD_COL - 1

#init the states grid

#this is the store the state value (of every cell) of the individual sweep
states = np.zeros(BOARD_ROW*BOARD_COL).reshape(BOARD_ROW,BOARD_COL)

#this is to store the cumulative value after the sweep ends
statesnew = np.zeros(BOARD_ROW*BOARD_COL).reshape(BOARD_ROW,BOARD_COL)

states[trow,tcol]=0 # terminal state value   
states[0,0]=0 # terminal state value

'''
Note on actions
The agent can move either up,down,left or right (4 possible actions)
From some cell on the border of the grid one or two actions may not be possible
as in the case of the top right and bottom left. All other edges have only 3 actions
Actions have equal probability of 0.25% each (0.25 * 4 = 100%)
The sum of probabilites of all actions has to be 100%

Each action has a reward of -1
'''

gamma=1
reward=-1 
print(states)
t=0
convergencelimit = 0.0001
converged = False

while not converged :
  row=0
  col=0
  c=1
  temp=0
  while c < (BOARD_ROW * BOARD_COL -1 ): #loop through all the cells of grid except first and last i.e 0 and 15
    row=int(c/BOARD_ROW)
    col=c-(row*BOARD_COL)
    
    #for each cell / state caluate the value for each action
    actionvalue=np.zeros(4)
    if row - 1 >=0: #top
      actionvalue[0]=0.25 * (reward + gamma*(states[row-1,col]))
    else:
      actionvalue[0]=0.25 *  (reward + gamma*(states[row,col]))
    
    if row + 1 < BOARD_ROW:#bottom
      actionvalue[1]=0.25 * (reward + gamma*(states[row+1,col]))
    else:
      actionvalue[1]=0.25 * (reward + gamma*(states[row,col]))
    
    if col - 1 >= 0:#left
      actionvalue[2]=0.25* (reward + gamma*(states[row,col-1]))
    else:
      actionvalue[2]=0.25 * (reward + gamma*(states[row,col]))
    
    if col + 1 < BOARD_COL: #right
      actionvalue[3]=0.25* (reward + gamma*(states[row,col+1]))
    else:
      actionvalue[3]=0.25 * (reward + gamma*(states[row,col]))
      
    # add up the value for each of the possible actions in the current
    # cell and update statesnew (Note: do not update the current grid values
    # until full sweep of the entire state space is done)
    
    statesnew[row,col]= np.sum(actionvalue)
    c=c+1
  
  
  #calculate the diff between the two state spaces
  diff = statesnew - states
  diffav = abs(np.sum(diff))/(BOARD_COL*BOARD_ROW)
  
  #after each pass update the state space with the new values  
  states = np.copy(statesnew)
  t=t+1  
  print(np.round(statesnew))

  #have the states converged (exit condition)
  if(diffav <= convergencelimit):
    break

print("Convergence reached after ", t , " steps")

'''
On convergence
+-----+-----+-----+-----+
|  0  | -14 | -20 | -22 |
+-----+-----+-----+-----+
| -14 | -18 | -20 | -20 |
+-----+-----+-----+-----+
| -20 | -20 | -18 | -14 |
+-----+-----+-----+-----+
| -22 | -20 | -14 |  0  |
+-----+-----+-----+-----+

'''




