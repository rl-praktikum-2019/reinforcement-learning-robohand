# Reinforcement-Learning-2019 - Training a robot hand to throw a ball

### Important Documentations

https://openai.github.io/mujoco-py/build/html/index.html (API of Physics Simulation)
http://www.mujoco.org/book/XMLreference.html (XML Reference - How to design a simulation environment via XML)
### Installation for Macbook
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

### Run and Smile :D

`python run_mujoco_test.py`
