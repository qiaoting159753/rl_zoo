import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
location = "/Users/tonyq/Desktop/CH1_Data/prior/overfit/Prior_World_Model_"


def load_pnn():
    algorithm = ""
    finger_1 = pd.read_csv(location + algorithm + "finger_turn_hard/0.1_16_16_15.csv.csv")
    finger_2 = pd.read_csv(location + algorithm + "finger_turn_hard/0.1_16_16_25.csv.csv")
    finger_3 = pd.read_csv(location + algorithm + "finger_turn_hard/0.1_16_16_35.csv.csv")
    reacher_1 = pd.read_csv(location + algorithm + "reacher_hard/0.1_16_16_15.csv.csv")
    reacher_2 = pd.read_csv(location + algorithm + "reacher_hard/0.1_16_16_25.csv.csv")
    reacher_3 = pd.read_csv(location + algorithm + "reacher_hard/0.1_16_16_35.csv.csv")
    fish_1 = pd.read_csv(location + algorithm + "fish_swim/0.1_16_16_15.csv.csv")
    fish_2 = pd.read_csv(location + algorithm + "fish_swim/0.1_16_16_25.csv.csv")
    fish_3 = pd.read_csv(location + algorithm + "fish_swim/0.1_16_16_35.csv.csv")
    hcheetah_1 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.1_16_16_15.csv.csv")
    hcheetah_2 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.1_16_16_25.csv.csv")
    hcheetah_3 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.1_16_16_35.csv.csv")
    hopper_1 = pd.read_csv(location + algorithm + "Hopper-v5_/0.1_16_16_15.csv.csv")
    hopper_2 = pd.read_csv(location + algorithm + "Hopper-v5_/0.1_16_16_25.csv.csv")
    hopper_3 = pd.read_csv(location + algorithm + "Hopper-v5_/0.1_16_16_35.csv.csv")
    swimmer_1 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.1_16_16_15.csv.csv")
    swimmer_2 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.1_16_16_25.csv.csv")
    swimmer_3 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.1_16_16_35.csv.csv")
    walker_1 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.1_16_16_15.csv.csv")
    walker_2 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.1_16_16_25.csv.csv")
    walker_3 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.1_16_16_35.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


def load_1():
    algorithm = ""
    finger_1 = pd.read_csv(location + algorithm + "finger_turn_hard/0.3_16_16_15.csv.csv")
    finger_2 = pd.read_csv(location + algorithm + "finger_turn_hard/0.3_16_16_25.csv.csv")
    finger_3 = pd.read_csv(location + algorithm + "finger_turn_hard/0.3_16_16_35.csv.csv")
    reacher_1 = pd.read_csv(location + algorithm + "reacher_hard/0.3_16_16_15.csv.csv")
    reacher_2 = pd.read_csv(location + algorithm + "reacher_hard/0.3_16_16_25.csv.csv")
    reacher_3 = pd.read_csv(location + algorithm + "reacher_hard/0.3_16_16_35.csv.csv")
    fish_1 = pd.read_csv(location + algorithm + "fish_swim/0.3_16_16_15.csv.csv")
    fish_2 = pd.read_csv(location + algorithm + "fish_swim/0.3_16_16_25.csv.csv")
    fish_3 = pd.read_csv(location + algorithm + "fish_swim/0.3_16_16_35.csv.csv")
    hcheetah_1 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.3_16_16_15.csv.csv")
    hcheetah_2 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.3_16_16_25.csv.csv")
    hcheetah_3 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.3_16_16_35.csv.csv")
    hopper_1 = pd.read_csv(location + algorithm + "Hopper-v5_/0.3_16_16_15.csv.csv")
    hopper_2 = pd.read_csv(location + algorithm + "Hopper-v5_/0.3_16_16_25.csv.csv")
    hopper_3 = pd.read_csv(location + algorithm + "Hopper-v5_/0.3_16_16_35.csv.csv")
    swimmer_1 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.3_16_16_15.csv.csv")
    swimmer_2 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.3_16_16_25.csv.csv")
    swimmer_3 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.3_16_16_35.csv.csv")
    walker_1 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.3_16_16_15.csv.csv")
    walker_2 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.3_16_16_25.csv.csv")
    walker_3 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.3_16_16_35.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


def load_2():
    algorithm = ""
    finger_1 = pd.read_csv(location + algorithm + "finger_turn_hard/0.5_16_16_15.csv.csv")
    finger_2 = pd.read_csv(location + algorithm + "finger_turn_hard/0.5_16_16_25.csv.csv")
    finger_3 = pd.read_csv(location + algorithm + "finger_turn_hard/0.5_16_16_35.csv.csv")
    reacher_1 = pd.read_csv(location + algorithm + "reacher_hard/0.5_16_16_15.csv.csv")
    reacher_2 = pd.read_csv(location + algorithm + "reacher_hard/0.5_16_16_25.csv.csv")
    reacher_3 = pd.read_csv(location + algorithm + "reacher_hard/0.5_16_16_35.csv.csv")
    fish_1 = pd.read_csv(location + algorithm + "fish_swim/0.5_16_16_15.csv.csv")
    fish_2 = pd.read_csv(location + algorithm + "fish_swim/0.5_16_16_25.csv.csv")
    fish_3 = pd.read_csv(location + algorithm + "fish_swim/0.5_16_16_35.csv.csv")
    hcheetah_1 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.5_16_16_15.csv.csv")
    hcheetah_2 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.5_16_16_25.csv.csv")
    hcheetah_3 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.5_16_16_35.csv.csv")
    hopper_1 = pd.read_csv(location + algorithm + "Hopper-v5_/0.5_16_16_15.csv.csv")
    hopper_2 = pd.read_csv(location + algorithm + "Hopper-v5_/0.5_16_16_25.csv.csv")
    hopper_3 = pd.read_csv(location + algorithm + "Hopper-v5_/0.5_16_16_35.csv.csv")
    swimmer_1 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.5_16_16_15.csv.csv")
    swimmer_2 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.5_16_16_25.csv.csv")
    swimmer_3 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.5_16_16_35.csv.csv")
    walker_1 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.5_16_16_15.csv.csv")
    walker_2 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.5_16_16_25.csv.csv")
    walker_3 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.5_16_16_35.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


def load_3():
    algorithm = ""
    finger_1 = pd.read_csv(location + algorithm + "finger_turn_hard/0.9_16_16_15.csv.csv")
    finger_2 = pd.read_csv(location + algorithm + "finger_turn_hard/0.9_16_16_25.csv.csv")
    finger_3 = pd.read_csv(location + algorithm + "finger_turn_hard/0.9_16_16_35.csv.csv")
    reacher_1 = pd.read_csv(location + algorithm + "reacher_hard/0.9_16_16_15.csv.csv")
    reacher_2 = pd.read_csv(location + algorithm + "reacher_hard/0.9_16_16_25.csv.csv")
    reacher_3 = pd.read_csv(location + algorithm + "reacher_hard/0.9_16_16_35.csv.csv")
    fish_1 = pd.read_csv(location + algorithm + "fish_swim/0.9_16_16_15.csv.csv")
    fish_2 = pd.read_csv(location + algorithm + "fish_swim/0.9_16_16_25.csv.csv")
    fish_3 = pd.read_csv(location + algorithm + "fish_swim/0.9_16_16_35.csv.csv")
    hcheetah_1 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.9_16_16_15.csv.csv")
    hcheetah_2 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.9_16_16_25.csv.csv")
    hcheetah_3 = pd.read_csv(location + algorithm + "HalfCheetah-v5_/0.9_16_16_35.csv.csv")
    hopper_1 = pd.read_csv(location + algorithm + "Hopper-v5_/0.9_16_16_15.csv.csv")
    hopper_2 = pd.read_csv(location + algorithm + "Hopper-v5_/0.9_16_16_25.csv.csv")
    hopper_3 = pd.read_csv(location + algorithm + "Hopper-v5_/0.9_16_16_35.csv.csv")
    swimmer_1 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.9_16_16_15.csv.csv")
    swimmer_2 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.9_16_16_25.csv.csv")
    swimmer_3 = pd.read_csv(location + algorithm + "Swimmer-v5_/0.9_16_16_35.csv.csv")
    walker_1 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.9_16_16_15.csv.csv")
    walker_2 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.9_16_16_25.csv.csv")
    walker_3 = pd.read_csv(location + algorithm + "Walker2d-v5_/0.9_16_16_35.csv.csv")
    data_list_pnn = [finger_1, reacher_1, fish_1, hcheetah_1, hopper_1, swimmer_1, walker_1,
                     finger_2, reacher_2, fish_2, hcheetah_2, hopper_2, swimmer_2, walker_2,
                     finger_3, reacher_3, fish_3, hcheetah_3, hopper_3, swimmer_3, walker_3]
    return data_list_pnn


# Control
attributes = ['steps', 'dyna_mse', 'reward_mse', 'dyna_corr_l2', 'dyna_corr_l1', 'reward_corr_l1', 'reward_gt_corr_l1']
envs = ["finger-turn_hard", "reacher-hard", "fish-swim", "HalfCheetah-v5", "Hopper-v5", "Swimmer-v5", "Walker2d-v5"]
title = envs[6]
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
        "Hopper-v5": hopper_pnn,
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
    label="lambda=0.1",
    errorbar="sd",
)

data_list_1 = load_1()
data_list_2 = load_2()
data_list_3 = load_3()

for j in range(3):
    if j == 0:
        data_list = data_list_1
        label_name = "0.3"
    if j == 1:
        data_list = data_list_2
        label_name = "0.5"
    if j == 2:
        data_list = data_list_3
        label_name = "0.9"

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
                "Hopper-v5": hopper_compare,
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
        label="lambda="+label_name,
        errorbar="sd",
    )
plt.ylim(-0.1, 1)
plt.style.use("seaborn-v0_8")
label_fontsize = 15
title_fontsize = 20
ticks_fontsize = 10
plt.grid()
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlabel("Steps", fontsize=label_fontsize)
plt.ylabel("Pearson-Correlation", fontsize=label_fontsize)
# plt.title(title, fontsize=title_fontsize)
plt.legend(loc="best").set_draggable(True)
plt.tight_layout(pad=0.5)
plt.savefig(title + "_" + attr + "prior_param.png")
plt.show()
