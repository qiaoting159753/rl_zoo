import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
location = "/Users/tonyq/Downloads/CH1_DATA/"


def load_pnn():
    finger_1 = pd.read_csv(location + "simple_pnn/Single_PNN_finger_turn_hard/16_16_16_15_24_11_13_12_07_14.csv.csv")
    reacher_1 = pd.read_csv(location + "simple_pnn/Single_PNN_reacher_hard/16_16_16_15_24_11_13_15_15_41.csv.csv")
    fish_1 = pd.read_csv(location + "simple_pnn/Single_PNN_fish_swim/16_16_16_15_24_11_13_13_27_31.csv.csv")
    hcheetah_1 = pd.read_csv(location + "simple_pnn/Single_PNN_HalfCheetah-v5_/16_16_16_15_24_11_15_09_26_27.csv.csv")
    hopper_1 = pd.read_csv(location + "simple_pnn/Single_PNN_Hopper-v5_/16_16_16_15_24_11_15_11_36_07.csv.csv")
    swimmer_1 = pd.read_csv(location + "simple_pnn/Single_PNN_Swimmer-v5_/16_16_16_15_24_11_15_09_00_42.csv.csv")
    walker_1 = pd.read_csv(location + "simple_pnn/Single_PNN_Walker2d-v5_/16_16_16_15_24_11_15_12_43_59.csv.csv")
    finger_2 = pd.read_csv(location + "simple_pnn/Single_PNN_finger_turn_hard/16_16_16_25_24_11_13_12_33_34.csv.csv")
    reacher_2 = pd.read_csv(location + "simple_pnn/Single_PNN_reacher_hard/16_16_16_25_24_11_13_15_47_23.csv.csv")
    fish_2 = pd.read_csv(location + "simple_pnn/Single_PNN_fish_swim/16_16_16_25_24_11_13_14_02_07.csv.csv")
    hcheetah_2 = pd.read_csv(location + "simple_pnn/Single_PNN_HalfCheetah-v5_/16_16_16_25_24_11_15_09_47_23.csv.csv")
    hopper_2 = pd.read_csv(location + "simple_pnn/Single_PNN_Hopper-v5_/16_16_16_25_24_11_15_11_58_44.csv.csv")
    swimmer_2 = pd.read_csv(location + "simple_pnn/Single_PNN_Swimmer-v5_/16_16_16_15_24_11_15_10_29_54.csv.csv")
    walker_2 = pd.read_csv(location + "simple_pnn/Single_PNN_Walker2d-v5_/16_16_16_25_24_11_15_13_06_38.csv.csv")
    finger_3 = pd.read_csv(location + "simple_pnn/Single_PNN_finger_turn_hard/16_16_16_35_24_11_13_13_00_09.csv.csv")
    reacher_3 = pd.read_csv(location + "simple_pnn/Single_PNN_reacher_hard/16_16_16_35_24_11_13_16_19_22.csv.csv")
    fish_3 = pd.read_csv(location + "simple_pnn/Single_PNN_fish_swim/16_16_16_35_24_11_13_14_39_28.csv.csv")
    hcheetah_3 = pd.read_csv(location + "simple_pnn/Single_PNN_HalfCheetah-v5_/16_16_16_35_24_11_15_10_08_36.csv.csv")
    hopper_3 = pd.read_csv(location + "simple_pnn/Single_PNN_Hopper-v5_/16_16_16_35_24_11_15_12_21_00.csv.csv")
    swimmer_3 = pd.read_csv(location + "simple_pnn/Single_PNN_Swimmer-v5_/16_16_16_25_24_11_15_10_51_59.csv.csv")
    walker_3 = pd.read_csv(location + "simple_pnn/Single_PNN_Walker2d-v5_/16_16_16_35_24_11_15_13_29_30.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


def load_prior():
    algorithm = "prior_network"
    finger_1 = pd.read_csv(location + algorithm + "/finger_turn_hard/0.3_16_16_15_24_11_18_04_47_57.csv.csv")
    finger_2 = pd.read_csv(location + algorithm + "/finger_turn_hard/0.3_16_16_25_24_11_18_05_09_44.csv.csv")
    finger_3 = pd.read_csv(location + algorithm + "/finger_turn_hard/0.3_16_16_35_24_11_18_05_31_09.csv.csv")
    reacher_1 = pd.read_csv(location + algorithm + "/reacher_hard/0.3_16_16_15_24_11_18_12_12_52.csv.csv")
    reacher_2 = pd.read_csv(location + algorithm + "/reacher_hard/0.3_16_16_25_24_11_18_12_36_55.csv.csv")
    reacher_3 = pd.read_csv(location + algorithm + "/reacher_hard/0.3_16_16_35_24_11_18_13_00_57.csv.csv")
    fish_1 = pd.read_csv(location + algorithm + "/fish_swim/0.3_16_16_15_24_11_18_08_01_05.csv.csv")
    fish_2 = pd.read_csv(location + algorithm + "/fish_swim/0.3_16_16_25_24_11_18_08_29_00.csv.csv")
    fish_3 = pd.read_csv(location + algorithm + "/fish_swim/0.3_16_16_35_24_11_18_08_56_56.csv.csv")
    hcheetah_1 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/0.3_16_16_15_24_11_18_04_43_35.csv.csv")
    hcheetah_2 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/0.3_16_16_15_24_11_18_04_47_39.csv.csv")
    hcheetah_3 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/0.3_16_16_25_24_11_18_05_09_02.csv.csv")
    hopper_1 = pd.read_csv(location + algorithm + "/Hopper-v5_/0.3_16_16_15_24_11_18_11_24_33.csv.csv")
    hopper_2 = pd.read_csv(location + algorithm + "/Hopper-v5_/0.3_16_16_25_24_11_18_11_47_18.csv.csv")
    hopper_3 = pd.read_csv(location + algorithm + "/Hopper-v5_/0.3_16_16_35_24_11_18_12_10_01.csv.csv")
    swimmer_1 = pd.read_csv(location + algorithm + "/Swimmer-v5_/0.3_16_16_15_24_11_18_08_02_22.csv.csv")
    swimmer_2 = pd.read_csv(location + algorithm + "/Swimmer-v5_/0.3_16_16_25_24_11_18_08_24_48.csv.csv")
    swimmer_3 = pd.read_csv(location + algorithm + "/Swimmer-v5_/0.3_16_16_35_24_11_18_08_47_10.csv.csv")
    walker_1 = pd.read_csv(location + algorithm + "/Walker2d-v5_/0.3_16_16_15_24_11_18_14_49_52.csv.csv")
    walker_2 = pd.read_csv(location + algorithm + "/Walker2d-v5_/0.3_16_16_25_24_11_18_15_13_03.csv.csv")
    walker_3 = pd.read_csv(location + algorithm + "/Walker2d-v5_/0.3_16_16_35_24_11_18_15_36_09.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


def load_ensemble():
    algorithm = "ensemble"
    finger_1 = pd.read_csv(location + algorithm + "/finger_turn_hard/15_0_0_15_24_11_13_22_52_20.csv.csv")
    finger_2 = pd.read_csv(location + algorithm + "/finger_turn_hard/15_0_0_25_24_11_13_23_50_40.csv.csv")
    finger_3 = pd.read_csv(location + algorithm + "/finger_turn_hard/15_0_0_35_24_11_14_00_53_36.csv.csv")
    reacher_1 = pd.read_csv(location + algorithm + "/reacher_hard/15_0_0_15_24_11_14_05_23_31.csv.csv")
    reacher_2 = pd.read_csv(location + algorithm + "/reacher_hard/15_0_0_25_24_11_14_06_29_56.csv.csv")
    reacher_3 = pd.read_csv(location + algorithm + "/reacher_hard/15_0_0_35_24_11_14_07_38_41.csv.csv")
    fish_1 = pd.read_csv(location + algorithm + "/fish_swim/15_0_0_15_24_11_14_01_55_07.csv.csv")
    fish_2 = pd.read_csv(location + algorithm + "/fish_swim/15_0_0_25_24_11_14_03_05_01.csv.csv")
    fish_3 = pd.read_csv(location + algorithm + "/fish_swim/15_0_0_35_24_11_14_04_13_53.csv.csv")
    hcheetah_1 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/15_0_0_15_24_11_12_23_26_29.csv.csv")
    hcheetah_2 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/15_0_0_25_24_11_13_00_23_27.csv.csv")
    hcheetah_3 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/15_0_0_35_24_11_13_01_23_03.csv.csv")
    hopper_1 = pd.read_csv(location + algorithm + "/Hopper-v5_/15_0_0_15_24_11_13_05_25_43.csv.csv")
    hopper_2 = pd.read_csv(location + algorithm + "/Hopper-v5_/15_0_0_25_24_11_13_06_27_04.csv.csv")
    hopper_3 = pd.read_csv(location + algorithm + "/Hopper-v5_/15_0_0_35_24_11_13_07_29_08.csv.csv")
    swimmer_1 = pd.read_csv(location + algorithm + "/Swimmer-v5_/15_0_0_15_24_11_13_02_22_05.csv.csv")
    swimmer_2 = pd.read_csv(location + algorithm + "/Swimmer-v5_/15_0_0_25_24_11_13_03_23_03.csv.csv")
    swimmer_3 = pd.read_csv(location + algorithm + "/Swimmer-v5_/15_0_0_35_24_11_13_04_24_42.csv.csv")
    walker_1 = pd.read_csv(location + algorithm + "/Walker2d-v5_/15_0_0_15_24_11_13_08_32_50.csv.csv")
    walker_2 = pd.read_csv(location + algorithm + "/Walker2d-v5_/15_0_0_25_24_11_13_09_32_00.csv.csv")
    walker_3 = pd.read_csv(location + algorithm + "/Walker2d-v5_/15_0_0_35_24_11_13_10_32_53.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


def load_bbb():
    algorithm = "bnn_bbb"
    finger_1 = pd.read_csv(location + algorithm + "/finger_turn_hard/5000_0.1_0.0_15_24_11_14_09_40_45.csv.csv")
    finger_2 = pd.read_csv(location + algorithm + "/finger_turn_hard/5000_0.1_0.0_25_24_11_14_10_00_27.csv.csv")
    finger_3 = pd.read_csv(location + algorithm + "/finger_turn_hard/5000_0.1_0.0_35_24_11_14_10_20_11.csv.csv")
    reacher_1 = pd.read_csv(location + algorithm + "/reacher_hard/5000_0.1_0.0_15_24_11_15_01_21_40.csv.csv")
    reacher_2 = pd.read_csv(location + algorithm + "/reacher_hard/5000_0.1_0.0_25_24_11_15_01_54_12.csv.csv")
    reacher_3 = pd.read_csv(location + algorithm + "/reacher_hard/5000_0.1_0.0_35_24_11_15_02_26_55.csv.csv")
    fish_1 = pd.read_csv(location + algorithm + "/fish_swim/5000_0.1_0.0_15_24_11_14_14_36_44.csv.csv")
    fish_2 = pd.read_csv(location + algorithm + "/fish_swim/5000_0.1_0.0_25_24_11_14_15_19_45.csv.csv")
    fish_3 = pd.read_csv(location + algorithm + "/fish_swim/5000_0.1_0.0_35_24_11_14_16_02_47.csv.csv")
    hcheetah_1 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/5000_0.1_0.0_15_24_11_14_09_37_27.csv.csv")
    hcheetah_2 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/5000_0.1_0.0_25_24_11_14_09_59_59.csv.csv")
    hcheetah_3 = pd.read_csv(location + algorithm + "/HalfCheetah-v5_/5000_0.1_0.0_35_24_11_14_10_22_32.csv.csv")
    hopper_1 = pd.read_csv(location + algorithm + "/Hopper-v5_/5000_0.1_0.0_15_24_11_14_22_00_31.csv.csv")
    hopper_2 = pd.read_csv(location + algorithm + "/Hopper-v5_/5000_0.1_0.0_25_24_11_14_22_19_02.csv.csv")
    hopper_3 = pd.read_csv(location + algorithm + "/Hopper-v5_/10000_0.1_0.0_15_24_11_14_23_02_01.csv.csv")
    swimmer_1 = pd.read_csv(location + algorithm + "/Swimmer-v5_/5000_0.1_0.0_15_24_11_14_15_16_19.csv.csv")
    swimmer_2 = pd.read_csv(location + algorithm + "/Swimmer-v5_/5000_0.1_0.0_25_24_11_14_15_43_20.csv.csv")
    swimmer_3 = pd.read_csv(location + algorithm + "/Swimmer-v5_/5000_0.1_0.0_35_24_11_14_16_10_18.csv.csv")
    walker_1 = pd.read_csv(location + algorithm + "/Walker2d-v5_/10000_0.1_0.0_15_24_11_15_02_55_43.csv.csv")
    walker_2 = pd.read_csv(location + algorithm + "/Walker2d-v5_/10000_0.1_0.0_25_24_11_15_03_18_08.csv.csv")
    walker_3 = pd.read_csv(location + algorithm + "/Walker2d-v5_/10000_0.1_0.0_35_24_11_15_03_40_21.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


# Control
attributes = ['steps', 'dyna_mse', 'reward_mse', 'dyna_corr_l2', 'dyna_corr_l1', 'reward_corr_l1', 'reward_gt_corr_l1']
envs = ["finger-turn_hard", "reacher-hard", "fish-swim", "HalfCheetah-v5", "Hopper-hop", "Swimmer-v5", "Walker2d-v5"]
title = envs[3]
attr = attributes[3]
cut = 1

data_list_pnn = load_pnn()
for d in data_list_pnn:
    d.columns = attributes

finger_pnn = pd.DataFrame([data_list_pnn[0][attr], data_list_pnn[7][attr], data_list_pnn[14][attr]])
reacher_pnn = pd.DataFrame([data_list_pnn[1][attr], data_list_pnn[8][attr], data_list_pnn[15][attr]])
fish_pnn = pd.DataFrame([data_list_pnn[2][attr], data_list_pnn[9][attr], data_list_pnn[16][attr]])
hcheetah_pnn = pd.DataFrame([data_list_pnn[3][attr], data_list_pnn[10][attr], data_list_pnn[17][attr]])
hopper_pnn = pd.DataFrame([data_list_pnn[4][attr], data_list_pnn[11][attr], data_list_pnn[18][attr]])
swimmer_pnn = pd.DataFrame([data_list_pnn[5][attr], data_list_pnn[12][attr], data_list_pnn[19][attr]])
walker_pnn = pd.DataFrame([data_list_pnn[6][attr], data_list_pnn[13][attr], data_list_pnn[20][attr]])

pnns = {"finger-turn_hard": finger_pnn,
        "reacher-hard": reacher_pnn,
        "fish-swim": fish_pnn,
        "HalfCheetah-v5": hcheetah_pnn,
        "Hopper-hop": hopper_pnn,
        "Swimmer-v5": swimmer_pnn,
        "Walker2d-v5": walker_pnn}

pnn_mean = pnns[title].mean(axis=0)
pnn_var = pnns[title].var(axis=0)
pnn_mean = pd.DataFrame({'steps': data_list_pnn[0]['steps'], 'Data': pnn_mean})
pnn_var = pd.DataFrame({'steps': data_list_pnn[0]['steps'], 'Data': pnn_var})


pnn_mean = pnn_mean.iloc[cut:, :]
pnn_var = pnn_var.iloc[cut:, :]
window_size = 10
pnn_mean["Data"] = pnn_mean["Data"].rolling(window_size, step=1, min_periods=1).mean()
sns.set_style("dark")
sns.lineplot(
    data=pnn_mean,
    x=pnn_mean["steps"],
    y="Data",
    label="pnn",
    errorbar="sd",
)

# plt.fill_between(pnn_mean["steps"],
#                  pnn_mean["Data"] - pnn_var["Data"],
#                  pnn_mean["Data"] + pnn_var["Data"],
#                  alpha=0.3)


data_list_1 = load_prior()
data_list_2 = load_ensemble()
data_list_3 = load_bbb()

for j in range(3):
    if j == 0:
        data_list = data_list_1
        label_name = "prior"
    if j == 1:
        data_list = data_list_2
        label_name = "ensemble"
    if j == 2:
        data_list = data_list_3
        label_name = "bayes backprop"

    for d in data_list:
        d.columns = attributes
    finger_compare = pd.DataFrame([data_list[0][attr], data_list[7][attr], data_list[14][attr]])
    reacher_compare = pd.DataFrame([data_list[1][attr], data_list[8][attr], data_list[15][attr]])
    fish_compare = pd.DataFrame([data_list[2][attr], data_list[9][attr], data_list[16][attr]])
    hcheetah_compare = pd.DataFrame([data_list[3][attr], data_list[10][attr], data_list[17][attr]])
    hopper_compare = pd.DataFrame([data_list[4][attr], data_list[11][attr], data_list[18][attr]])
    swimmer_compare = pd.DataFrame([data_list[5][attr], data_list[12][attr], data_list[19][attr]])
    walker_compare = pd.DataFrame([data_list[6][attr], data_list[13][attr], data_list[20][attr]])
    compares = {"finger-turn_hard": finger_compare,
                "reacher-hard": reacher_compare,
                "fish-swim": fish_compare,
                "HalfCheetah-v5": hcheetah_compare,
                "Hopper-hop": hopper_compare,
                "Swimmer-v5": swimmer_compare,
                "Walker2d-v5": walker_compare,
                }
    compare_mean = compares[title].mean(axis=0)
    compare_var = compares[title].var(axis=0)
    compare_mean = pd.DataFrame({'steps': data_list_pnn[0]['steps'], 'Data': compare_mean})
    compare_var = pd.DataFrame({'steps': data_list_pnn[0]['steps'], 'Data': compare_var})

    # att_name = 'dyna_mse'
    window_size = 10

    compare_mean["Data"] = compare_mean["Data"].rolling(window_size, step=1, min_periods=1).mean()
    compare_mean = compare_mean.iloc[cut:, :]
    compare_var = compare_var.iloc[cut:, :]

    ################################    PPPPPPLLLLLLOOOOOOTTTTTT    ###########
    sns.lineplot(
        data=compare_mean,
        x=compare_mean["steps"],
        y="Data",
        label=label_name,
        errorbar="sd",
    )
    # plt.fill_between(compare_mean["steps"],
    #                  compare_mean["Data"] - compare_var["Data"],
    #                  compare_mean["Data"] + compare_var["Data"],
    #                 alpha=0.3)

plt.style.use("seaborn-v0_8")
label_fontsize = 15
title_fontsize = 20
ticks_fontsize = 10
plt.grid()
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlabel("Steps", fontsize=label_fontsize)
plt.ylabel("Pearson-Correlation", fontsize=label_fontsize)
plt.title(title, fontsize=title_fontsize)
plt.legend(loc="best").set_draggable(True)
plt.tight_layout(pad=0.5)
plt.savefig(title + "_" + attr + ".png")
plt.show()









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
