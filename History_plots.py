import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.pyplot import cm
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

model_headers = ["wall", "epoch", "loss", "value"]

# Depth
model_numbers = [1, 2, 3, 4, 5, 6, 7, 8]
plt.figure(figsize=(10, 6))
colour = iter(cm.rainbow(np.linspace(0, 1, len(model_numbers))))
for num in model_numbers:
    c = next(colour)
    model = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_"+str(num)+"x5_mask_2899_LRELU_int_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    model2 = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_"+str(num)+"x5_mask_2899_RELU_int_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    plt.plot(model["epoch"], model['loss'], label=f"Depth: {str(num)}", c=c)
    plt.plot(model2["epoch"], model2['loss'], c=c, alpha=0.4, linestyle='dashed')

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation loss (MAE)")
plt.ylim([0.05, 0.2])
plt.xlim([0, 350])
plt.savefig("Data/Data_for_ML/Model_history/Learning_depth.pdf")
plt.show()

# Width
model_numbers = [2, 5, 1]
name = [200, 512, 1000]
i = 0
plt.figure(figsize=(10, 6))
colour = iter(cm.rainbow(np.linspace(0, 1, len(model_numbers))))
for num in model_numbers:
    c = next(colour)
    model = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_2x"+str(num)+"_mask_2899_LRELU_int_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    model2 = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_2x"+str(num)+"_mask_2899_RELU_int_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    plt.plot(model["epoch"], model['loss'], label=f"Width: {name[i]}", c=c)
    plt.plot(model2["epoch"], model2['loss'], c=c, linestyle='dashed')
    i += 1
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation loss (MAE)")
plt.ylim([0.05, 0.2])
plt.xlim([0, 350])
plt.savefig("Data/Data_for_ML/Model_history/Learning_width.pdf")
plt.show()

# Activation
model_names = ["LRELU", "RELU", "ELU", "linear", "sigmoid", "tanh"]
name = ["Leaky ReLU", "ReLU", "ELU", "Linear", "Sigmoid", "Tanh"]
i = 0
fig, ax = plt.subplots(figsize=(10, 6))
rect = [0.29, 0.45, 0.4, 0.4]  # [left, bottom, width, height]
ax_zoomed = fig.add_axes(rect)
for num in model_names:
    model = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_2x5_mask_2899_"+str(num)+"_int_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    ax.plot(model["epoch"], model['loss'], label=f"{name[i]}")
    ax_zoomed.plot(model["epoch"], model['loss'], label=f"{name[i]}")
    i += 1

rectangle = patches.Rectangle((0.0, 0.065), 350, 0.185, linewidth=1, edgecolor='r',
                              linestyle='dashed', facecolor='none')
ax_zoomed.add_patch(rectangle)

ax.legend()
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation loss (MAE)")

ax_zoomed.set_ylim([0.06, 0.4])
ax_zoomed.set_xlim([0, 350])
# ax_zoomed.yaxis.set_ticks([])
ax_zoomed.xaxis.set_ticks([])

ax.set_ylim([0.065, 0.25])
ax.set_xlim([0, 350])
plt.savefig("Data/Data_for_ML/Model_history/Learning_activation.pdf")
plt.show()
