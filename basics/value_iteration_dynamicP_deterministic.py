# test

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

# trow, tcol define the winning/target cell
trow = BOARD_ROW -1 
tcol = BOARD_COL - 1

#init the states grid
states = np.zeros(BOARD_ROW*BOARD_COL).reshape(BOARD_ROW,BOARD_COL)
statesnew = np.zeros(BOARD_ROW*BOARD_COL).reshape(BOARD_ROW,BOARD_COL)
#states[states==0]=-1 # set the initial state as all -1 
states[trow,tcol]=0 # set value of target cell as +1 # except this terminal state (top left)
states[0,0]=0 #and this termianl state bottom right

gamma=1
reward=-1
print(states)
t=0
while t < 24:
  row=0
  col=0
  c=1
  temp=0
  while c < (BOARD_ROW * BOARD_COL -1 ): #loop through all the cells of grid except first and last i.e 0 and 15
    row=int(c/BOARD_ROW)
    col=c-(row*BOARD_COL)
    #print("[count] r,c",c,row,col)
    
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
      
    statesnew[row,col]= np.sum(actionvalue)
    c=c+1
  
  
  #copy the new states value after scanning the grid once  
  t=t+1  
  print(statesnew)


