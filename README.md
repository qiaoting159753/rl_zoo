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
| DQN |               Yes | --- | --- |
| Double DQN |        Yes | --- | --- |
| Dueling DQN |       Yes | --- | --- |
| Distributional    | Yes | --- | --- |
| Rainbow           | Yes | --- | --- |
| Policy Gradient (PG) | Yes | --- | --- |
| Deep Deterministic PG | Yes | --- | --- |
| TD-3               | Yes | --- | --- |
| Soft Actor  Critic | Yes | --- | --- |
