# rl_zoo
I aim to include as many algorithms as possible in Model-Free Reinforcement Learning and Model-Based Reinforcement learning. However, the primary focus will be on Model-Based Reinforcement Learning.

## How to use it. 
### Installation 
```
git clone https://github.com/qiaoting159753/rl_zoo.git
cd rl_zoo
```
```
pip install rl_zoo
```
### Run from a Command Line 
```
python3 main.py --env_config=PATH_TO_ENV_CONFIG --agent_config=PATH_TO_AGENT_CONFIG --train_config=PATH_TO_TRAIN_CONFIG
```

## Algorithm included.
| Name | Discrete/Continuous | Model-Free | Model-Based |
| --- |               --- | --- | --- |
| Deep Q Network (DQN)| Discrete | --- | --- |
| Double DQN |        Discrete | --- | --- |
| Dueling DQN |       Discrete | --- | --- |
| Distributional DQN| Discrete | --- | --- |
| Rainbow           | Discrete | --- | --- |
| Policy Gradient (PG)| Continuous | --- | --- |
| Deep Deterministic PG | Continuous | --- | --- |
| TD-3              | Continuous | --- | --- |
| Soft Actor Critic | Continuous | --- | --- |
| Trust Region Policy Optimization | Continuous | --- | --- |
| Proximal Policy Optimization | Continuous | --- | --- |

