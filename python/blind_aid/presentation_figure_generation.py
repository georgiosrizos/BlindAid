########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib import colors as mcolors

sns.set_style("darkgrid")
sns.set_context("paper")


if __name__ == "__main__":
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    ####################################################################################################################
    # Position figure.
    ####################################################################################################################

    f, axes = plt.subplots(1, 1, figsize=(9, 9))
    palette_seed = np.linspace(0, 3, 10)
    palette_counter = 0

    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes.add_patch(patches.Rectangle((0, 0),
                                     3,
                                     3,
                                     fill=False,
                                     linewidth=1))
    axes.scatter(np.array([0, 0, 0, 0,
                           1, 1, 1, 1,
                           2, 2, 2, 2,
                           3, 3, 3, 3]),
                 np.array([0, 1, 2, 3,
                           0, 1, 2, 3,
                           0, 1, 2, 3,
                           0, 1, 2, 3]), s=100, marker="h", color=colors["salmon"])
    axes.set(xlim=(-1, 4), ylim=(-1, 4))
    axes.set_xlabel("metres", size=20)
    axes.set_ylabel("metres", size=20)
    axes.tick_params(labelsize=9)
    palette_counter += 1

    f.tight_layout()
    plt.show()

    ####################################################################################################################
    # Trajectory figure.
    ####################################################################################################################

    f, axes = plt.subplots(2, 2, figsize=(9, 9))
    palette_seed = np.linspace(0, 3, 10)
    palette_counter = 0

    axes[0, 0].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[0, 1].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[1, 0].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[1, 1].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))

    trajectories = list()

    noise = [0.1, 0.05, 0.2, 0.15]
    for i in range(4):
        trajectories.append((np.concatenate([np.ones(20, dtype=np.float64) * 0.2 + np.random.randn(20) * noise[i],
                                             np.linspace(0.2, 2.8, 20) + np.random.randn(20) * noise[i],
                                             np.ones(20, dtype=np.float64) * 2.8 + np.random.randn(20) * noise[i],
                                             np.linspace(2.8, 1.02105263, 14) + np.random.randn(14) * noise[i]]),
                             np.concatenate([np.linspace(0.2, 2.8, 20) + np.random.randn(20) * noise[i],
                                             np.ones(20, dtype=np.float64) * 2.8 + np.random.randn(20) * noise[i],
                                             np.linspace(2.8, 0.2, 20) + np.random.randn(20) * noise[i],
                                             np.ones(14, dtype=np.float64) * 0.2 + np.random.randn(14) * noise[i]])))

    axes[0, 0].scatter(trajectories[0][0],
                       trajectories[0][1], marker=".", color=colors["red"])
    axes[0, 0].set(xlim=(-1, 4), ylim=(-1, 4))
    axes[0, 0].set_title("trial #1", fontsize=20)
    axes[0, 0].set_ylabel("metres", size=20)
    axes[0, 0].tick_params(labelsize=9)
    palette_counter += 1

    axes[0, 1].scatter(trajectories[1][0],
                       trajectories[1][1], marker=".", color=colors["red"])
    axes[0, 1].set(xlim=(-1, 4), ylim=(-1, 4))
    axes[0, 1].set_title("trial #2", fontsize=20)
    axes[0, 1].tick_params(labelsize=9)
    palette_counter += 1

    axes[1, 0].scatter(trajectories[2][0],
                       trajectories[2][1], marker=".", color=colors["red"])
    axes[1, 0].set(xlim=(-1, 4), ylim=(-1, 4))
    axes[1, 0].set_title("trial #3", fontsize=20)
    axes[1, 0].set_xlabel("metres", size=20)
    axes[1, 0].set_ylabel("metres", size=20)
    axes[1, 0].tick_params(labelsize=9)
    palette_counter += 1

    axes[1, 1].scatter(trajectories[3][0],
                       trajectories[3][1], marker=".", color=colors["red"])
    axes[1, 1].set(xlim=(-1, 4), ylim=(-1, 4))
    axes[1, 1].set_title("trial #4", fontsize=20)
    axes[1, 1].set_xlabel("metres", size=20)
    axes[1, 1].tick_params(labelsize=9)
    palette_counter += 1

    f.tight_layout()
    plt.show()
