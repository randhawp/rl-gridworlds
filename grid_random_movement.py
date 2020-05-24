import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import random

def redraw(img):
    img = env.render('rgb_array', tile_size=10)
    window.show_img(img)

def reset():
    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    print(obs)
    if done:
        print('done!')
        return -1
    else:
        redraw(obs)
    return 0

def random_solve(event):
    print('pressed', event.key)
    cnt=0;
    while cnt < 100:
        num = random.randint(0, 3)
        if num==0:
            r=step(env.actions.forward)
            
            
        if num==1:
            r=step(env.actions.left)
            
            
        if num==2:
            r=step(env.actions.right)
            
            
        if num==3:
            r=step(env.actions.forward)
            
            
        cnt=cnt+1
        if r== -1:
            reset()
            break

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return



env = gym.make('MiniGrid-Empty-5x5-v0')

env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

window = Window('gym_minigrid')
window.reg_key_handler(random_solve)

reset()

# Blocking event loop
window.show(block=True)
