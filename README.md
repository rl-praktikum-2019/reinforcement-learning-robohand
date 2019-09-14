# LMU Reinforcement Learning Praktikum 2019
## Training a robot hand to throw a ball

### Important Documentations

https://openai.github.io/mujoco-py/build/html/index.html (API of Physics Simulation)

http://www.mujoco.org/book/XMLreference.html (XML Reference - How to design a simulation environment via XML)

https://github.com/openai/gym/tree/master/gym/wrappers (Gym Wrapper to extend gym code functionality)

# Installation

## Installation for Macbook
```
conda create reinforcement_learning_venv python=3.6
conda activate reinforcement_learning_venv

pip install gym

brew install cmake boost boost-python sdl2 swig wget

brew install gcc
ls zu gcc-x in '/usr/local/Cellar/gcc'
export CC=/usr/local/Cellar/gcc/9.1.0/bin/gcc-9

pip install -U 'mujoco-py<2.1,>=2.0'
````
### Environment variables for conda env
```
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
````
Add following exports to $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh:
```
export LD_LIBRARY_PATH=/Users/studyingam//.mujoco
export PYTHONPATH="$PYTHONPATH:/Users/studyingam/git_tree/reinforcement-learning-2019"
````



## Installation on Ubuntu

Get your mujoco license here by following the listed steps: https://www.roboti.us/license.html

Download your desired MuJoCo binaries (not HAPTIX or Plugins) https://www.roboti.us/index.html

Unzip to ``$HOME/.mujoco/mjpro1XXX/`` where XXX is the version number f.e. 200 for 2.0

Add your key text file as described in the email from mujoco under ``$HOME/.mujoco/``.

Add your mujoco bin path to to end of your .bashrc via:
```
nano ~/.bashrc
then add: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujocoXXX/bin
source ~/.bashrc
````

Run the following command to install mujoco-py

(Hint: If the PyCharm terminal can not find the package -> run the command from the ubuntu terminal targeting the python.exe from your desired (conda) environment)

```
pip install mujoco-py
pip install gym
````

# Run

## Ubuntu with PyCharm
Add ``LD_LIBRARY_PATH`` with value ``$HOME/.mujoco/mujocoXXX/bin`` as above as a environment variable to your run configuration of the main execution .py file.

# Custom environment for ball throwing
Before installing the custom env, make sure gym has been uninstalled.

Install custom environment via: 
``pip install git+git://github.com/rl-praktikum-2019/gym.git@throw-ball-environment``

Extends the given HandEnv for gym by:

- Removing the visible target goal/ball which was floating above the hand and obstructed a good view on hand and ball
- Reduced mass of the ball in order to enable higher throws
- Added a visible ground plane to give more depth to the scene
- Changed reward in order to encourage ball throwing in vertical direction.

## Run commands: 

- E.g.: For ddpg: 'python train_main.py --render-env --env=ThrowBall-v0' --method='ddpg'
