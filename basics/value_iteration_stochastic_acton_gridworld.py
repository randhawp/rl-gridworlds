'''
A minimilist program to show value iteration in a deterministic grid world
v(s) <- v(s)t + learning_rate * ( v(s)t+1 - v(st))


There is only one reward +1 at the end of the episode
There is no other negative rewards
The actions in deterministic world are equally likley with a 25% chance
In stochastic the actions can have different probability
'''

import numpy as np 
import pandas 
import random

BOARD_ROW = 4
BOARD_COL = 4

START_ROW=0
START_COL=0
current_row=START_ROW
current_col=START_COL

row_labels = np.arange(0, BOARD_ROW, 1).astype('S')
column_labels = np.arange(0, BOARD_COL, 1).astype('S')

# prow, pcol define the winning/target cell
trow = BOARD_ROW -1 
tcol = BOARD_COL -1


states = np.zeros(BOARD_ROW*BOARD_COL).reshape(BOARD_ROW,BOARD_COL)
states[trow,tcol]=1 # set value of target cell as +1

#agents actions of movement in the grid are as below
actions = ["up", "down", "left", "right"]

#for each run we need to store the cells traversed to react the target cell
MAX_EPISODES = 14
episodelen = 0

while episodelen < MAX_EPISODES:
  episodepath = []

  while 1: #keep looping till the agent reaches target cell
    
    #0 -up, 1-down, 2-left,3-right
    
    choice = np.random.choice([0,1,2,3], p=[0.04, 0.03, 0.9,0.03])
    
    if actions[choice] == "up" and current_row-1 > 0:
      current_row = current_row - 1
      episodepath.append(current_row*BOARD_ROW + current_col)  
    if actions[choice] == "down" and current_row+1 < BOARD_ROW:
      current_row = current_row + 1
      episodepath.append(current_row*BOARD_ROW + current_col)  
    if actions[choice] == "left" and current_col-1 > 0:
      current_col = current_col - 1
      episodepath.append(current_row*BOARD_ROW + current_col)  
    if actions[choice] == "right" and current_col+1 < BOARD_COL:
      current_col = current_col + 1
      episodepath.append(current_row*BOARD_ROW + current_col)  
    if(current_col ==  tcol) and (current_row == trow):
      break
    
  # value iteration - going back from the target to the start along the path 
  # taken by the agent and updating the values of all cells  
  prow=trow
  pcol=tcol
  for n in episodepath[::-1]:
    row=int(n/BOARD_ROW)
    col=n-(row*BOARD_COL)
    #print("[n] r,c",n,row,col)
    states[row,col]= states[row,col] + 0.2*(states[prow,pcol] - states[row,col] )
    prow=row
    pcol=col

  #reset the episode and start a new one, send the agent back to start cell
  current_col=START_COL
  current_row=START_ROW
  
  print(episodepath)
  episodelen=episodelen+1
  print("done cycle ",episodelen)

  df = pandas.DataFrame(states, columns=column_labels, index=row_labels).round(3)
  print(df)