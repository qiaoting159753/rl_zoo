# rl_zoo
I aim to include as many algorithms as possible in Model-Free Reinforcement Learning and Model-Based Reinforcement learning. However, the primary focus will be on Model-Based Reinforcement Learning. Trick to improve data efficiency: do as many as updates you can with the collected data. Do regularization: Ensemble of Q/Model, Norm Layer. 

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
### Evaluation Benchmarks 
OpenAI Gymnasium. 
DeepMind Control Suite.

## Algorithm included.
| Name | Discrete/Continuous | Model-Free | Model-Based | Paper |
| ---                                | ---        | --- | --- | --- |
| Deep Q Network (DQN)               | Discrete   | Yes | No  | --- |
| Double DQN                         | Discrete   | Yes | No  | --- |
| Dueling DQN                        | Discrete   | Yes | No  | --- |
| Distributional DQN                 | Discrete   | Yes | No  | --- |
| Rainbow                            | Discrete   | Yes | No  | --- |
| Policy Gradient (PG)               | Continuous | Yes | No  | --- |
| Deep Deterministic PG              | Continuous | Yes | No  | --- |
| TD-3                               | Continuous | Yes | No  | --- |
| Soft Actor Critic                  | Continuous | Yes | No  | --- |
| Trust Region Policy Optimization   | Continuous | Yes | No  | --- |
| Proximal Policy Optimization       | Continuous | Yes | No  | --- |
| Dyna                               | ---        | No  | Yes | --- |
| Model-Based Value Expansion-Actor  | ---        | No  | Yes | --- |
| Model-Based Value Expansion-Critic | ---        | No  | Yes | --- |
| STEVE                              | ---        | No  | Yes | --- |
| PILCO                              | ---        | No  | Yes | --- |

