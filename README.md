# LMU Reinforcement Learning Praktikum 2019
## Training a robot hand to throw a ball

### Important Documentations

https://openai.github.io/mujoco-py/build/html/index.html (API of Physics Simulation)
http://www.mujoco.org/book/XMLreference.html (XML Reference - How to design a simulation environment via XML)

# Installation

## Installation for Macbook
```
conda create mujoco_venv python=3.6 anaconda
conda activate mujoco_venv

pip install gym

brew install cmake boost boost-python sdl2 swig wget

brew install gcc
ls zu gcc-x in '/usr/local/Cellar/gcc'
export CC=/usr/local/Cellar/gcc/9.1.0/bin/gcc-9

pip install -U 'mujoco-py<2.1,>=2.0'
````
## Installation on Ubuntu
Add your mujoco bin path (export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/patrick/.mujoco/mujocoXXX/bin) to to end of your .bashrc via:
```
nano ~/.bashrc
source ~/.bashrc
````
Add ``LD_LIBRARY_PATH`` with value as above as a environment variable to your run configuration of the main execution .py file.

Run the following command from the terminal using the python.exe from your desired (conda) environment
```
pip install mujoco-py
````

# Run
