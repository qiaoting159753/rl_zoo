import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)

title = "cheetah-run"

# plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_finger_turn_hard/16_16_16_15_24_10_09_22_51_05.csv.csv")
# plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_reacher_hard/16_16_16_15_24_10_09_21_45_55.csv.csv")
plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_fish_swim/16_16_16_15_24_10_10_00_21_26.csv.csv")


# plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_acrobot_swingup/16_16_16_15_24_10_09_23_13_16.csv.csv")
# plot_frame_prior =    pd.read_csv("/Users/tonyq/Desktop/ready_data/Prior_World_Model_acrobot_swingup/1.0_16_16_15_24_10_08_03_13_33.csv.csv")
# plot_frame_ensemble = pd.read_csv("/Users/tonyq/Desktop/ready_data/Ensemble_Dyna_Ensemble_Reward_acrobot_swingup/15_0_0_15_24_10_09_07_47_04.csv.csv")
# plot_frame_vi =       pd.read_csv("/Users/tonyq/Desktop/ready_data/Bayesian_VI_acrobot_swingup/50000_0.1_0.0_15_24_10_10_07_55_34.csv.csv")

plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_cheetah_run/16_16_16_15_24_10_09_21_02_47.csv.csv")
plot_frame_prior =    pd.read_csv("/Users/tonyq/Desktop/ready_data/Prior_World_Model_cheetah_run/1.0_16_16_15_24_10_07_23_49_36.csv.csv")
plot_frame_ensemble = pd.read_csv("/Users/tonyq/Desktop/ready_data/Ensemble_Dyna_Ensemble_Reward_cheetah_run/15_0_0_15_24_10_09_04_42_44.csv.csv")
plot_frame_vi =       pd.read_csv("/Users/tonyq/Desktop/ready_data/Bayesian_VI_cheetah_run/0.9_0.9_0.9_15_24_10_11_21_10_49.csv.csv")

# plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_acrobot_swingup/16_16_16_15_24_10_09_23_13_16.csv.csv")
# plot_frame_prior =    pd.read_csv("/Users/tonyq/Desktop/ready_data/Prior_World_Model_acrobot_swingup/1.0_16_16_15_24_10_08_03_13_33.csv.csv")
# plot_frame_ensemble = pd.read_csv("/Users/tonyq/Desktop/ready_data/Ensemble_Dyna_Ensemble_Reward_acrobot_swingup/15_0_0_15_24_10_09_07_47_04.csv.csv")
# plot_frame_vi =       pd.read_csv("/Users/tonyq/Desktop/ready_data/Bayesian_VI_acrobot_swingup/50000_0.1_0.0_15_24_10_10_07_55_34.csv.csv")
#
# plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_acrobot_swingup/16_16_16_15_24_10_09_23_13_16.csv.csv")
# plot_frame_prior =    pd.read_csv("/Users/tonyq/Desktop/ready_data/Prior_World_Model_acrobot_swingup/1.0_16_16_15_24_10_08_03_13_33.csv.csv")
# plot_frame_ensemble = pd.read_csv("/Users/tonyq/Desktop/ready_data/Ensemble_Dyna_Ensemble_Reward_acrobot_swingup/15_0_0_15_24_10_09_07_47_04.csv.csv")
# plot_frame_vi =       pd.read_csv("/Users/tonyq/Desktop/ready_data/Bayesian_VI_acrobot_swingup/50000_0.1_0.0_15_24_10_10_07_55_34.csv.csv")
#
# plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_acrobot_swingup/16_16_16_15_24_10_09_23_13_16.csv.csv")
# plot_frame_prior =    pd.read_csv("/Users/tonyq/Desktop/ready_data/Prior_World_Model_acrobot_swingup/1.0_16_16_15_24_10_08_03_13_33.csv.csv")
# plot_frame_ensemble = pd.read_csv("/Users/tonyq/Desktop/ready_data/Ensemble_Dyna_Ensemble_Reward_acrobot_swingup/15_0_0_15_24_10_09_07_47_04.csv.csv")
# plot_frame_vi =       pd.read_csv("/Users/tonyq/Desktop/ready_data/Bayesian_VI_acrobot_swingup/50000_0.1_0.0_15_24_10_10_07_55_34.csv.csv")
#
# plot_frame_pnn =      pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_acrobot_swingup/16_16_16_15_24_10_09_23_13_16.csv.csv")
# plot_frame_prior =    pd.read_csv("/Users/tonyq/Desktop/ready_data/Prior_World_Model_acrobot_swingup/1.0_16_16_15_24_10_08_03_13_33.csv.csv")
# plot_frame_ensemble = pd.read_csv("/Users/tonyq/Desktop/ready_data/Ensemble_Dyna_Ensemble_Reward_acrobot_swingup/15_0_0_15_24_10_09_07_47_04.csv.csv")
# plot_frame_vi =       pd.read_csv("/Users/tonyq/Desktop/ready_data/Bayesian_VI_acrobot_swingup/50000_0.1_0.0_15_24_10_10_07_55_34.csv.csv")


# plot_frame_la =       pd.read_csv("/Users/tonyq/Desktop/ready_data/Single_PNN_acrobot_swingup/16_16_16_15_24_10_09_23_13_16.csv.csv")

attributes = ['steps', 'dyna_mse', 'reward_mse', 'dyna_corr_l2', 'dyna_corr_l1', 'N/A', 'N/A', 'reward_corr_l1', 'N/A']
plot_frame_pnn.columns = attributes
plot_frame_prior.columns = attributes
plot_frame_ensemble.columns = attributes
plot_frame_vi.columns = attributes

plt.style.use("seaborn-v0_8")
label_fontsize = 15
title_fontsize = 20
ticks_fontsize = 10
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlabel("Steps", fontsize=label_fontsize)
plt.ylabel("MSE", fontsize=label_fontsize)


plt.title(title, fontsize=title_fontsize)

# att_name = 'dyna_mse'
att_name = 'dyna_corr_l1'

window_size = 10

plot_frame_pnn[att_name] = plot_frame_pnn[att_name].rolling(window_size, step=1, min_periods=1).mean()
sns.lineplot(
    data=plot_frame_pnn,
    x=plot_frame_pnn["steps"],
    y=att_name,
    label="pnn",
    errorbar="sd",
)

# plot_frame_prior[att_name] = plot_frame_prior[att_name].rolling(window_size, step=1, min_periods=1).mean()
# sns.lineplot(
#     data=plot_frame_prior,
#     x=plot_frame_prior["steps"],
#     y=att_name,
#     label="prior",
#     errorbar="sd",
# )

# plot_frame_ensemble[att_name] = plot_frame_ensemble[att_name].rolling(window_size, step=1, min_periods=1).mean()
# sns.lineplot(
#     data=plot_frame_ensemble,
#     x=plot_frame_ensemble["steps"],
#     y=att_name,
#     label='ensemble',
#     errorbar="sd",
# )

# plot_frame_vi[att_name] = plot_frame_vi[att_name].rolling(window_size, step=1, min_periods=1).mean()
# sns.lineplot(
#     data=plot_frame_vi,
#     x=plot_frame_vi["steps"],
#     y=att_name,
#     label='variational inference',
#     errorbar="sd",
# )



plt.legend(loc="best").set_draggable(True)
plt.tight_layout(pad=0.5)
plt.show()

# if not os.path.exists(f"{directory}/figures"):
#     os.makedirs(f"{directory}/figures")
# plt.savefig(f"{directory}/figures/{filename}.png")






# def prepare_train_plot_frame(
#     train_data: pd.DataFrame, window_size: int
# ) -> pd.DataFrame:
#     x_data: str = "total_steps"
#     y_data: str = "episode_reward"
#     plot_frame: pd.DataFrame = pd.DataFrame()
#     plot_frame["steps"] = train_data[x_data]
#     plot_frame["avg"] = (
#         train_data[y_data].rolling(window_size, step=1, min_periods=1).mean()
#     )
#     plot_frame["std_dev"] = (
#         train_data[y_data].rolling(window_size, step=1, min_periods=1).std()
#     )
#     return plot_frame










# def plot_train(
#     train_data: pd.DataFrame,
#     title: str,
#     label: str,
#     directory: str,
#     filename: str,
#     window_size: int,
#     display: bool = False,
# ) -> None:
#     train_plot_frame = prepare_train_plot_frame(train_data, window_size)
#     plot_data(
#         train_plot_frame,
#         title,
#         label,
#         "Steps",
#         "Average Reward",
#         directory,
#         filename,
#         display=display,
#     )


# def get_param_value(param_tag: str, config: dict) -> str:
#     if param_tag in config:
#         return config[param_tag]
#     return None


# def get_param_tag(param_tags: dict, alg_config: dict, train_config: dict) -> str:
#     if len(param_tags) == 0:
#         return ""
#     param_tag = ""
#     for key, tags in param_tags.items():
#         value = get_param_value(key, alg_config)
#         if value is None:
#             value = get_param_value(key, train_config)
#         if isinstance(value, dict):
#             tags = tags.split(",")
#             for tag in tags:
#                 tag = tag.strip()
#                 secondary_value = get_param_value(tag, value)
#                 if secondary_value is not None:
#                     param_tag += f"_{tag}_{secondary_value}"
#         elif value is not None:
#             param_tag += f"_{key}_{value}"
#     return param_tag


# def generate_labels(
#     args, title: str, result_directory: str
# ) -> tuple[str, str, str, str]:
#     env_config = read_environmnet_config(result_directory)
#     train_config = read_train_config(result_directory)
#     alg_config = read_algorithm_config(result_directory)
#
#     algorithm = alg_config["algorithm"]
#     domain = env_config["domain"]
#     task = env_config["task"]
#     task = task if domain == "" else f"{domain}-{task}"
#
#     param_tag = get_param_tag(args["param_tag"], alg_config, train_config)
#     label = algorithm + param_tag
#
#     title = task if title == "" else title
#     return title, algorithm, task, label

