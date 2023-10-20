import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

model_headers = ["wall", "epoch", "loss", "value"]

# Depth
model_numbers = [5, 55, 555, 5555, 55555, 555555, 5555555]
plt.figure(figsize=(10, 6))
for num in model_numbers:
    model = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_"+str(num)+"_mask_900_LRELU_int_up_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    plt.plot(model["epoch"], model['loss'], label=f"Depth: {len(str(num))}")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation loss (MAE)")
plt.ylim([0.1, 0.2])
plt.xlim([0, 600])
# plt.savefig("Data/Data_for_ML/Model_history/Learning_depth.pdf")
plt.show()

# Width
model_numbers = [200, 555, 1000]
name = [200, 512, 1000]
i = 0
plt.figure(figsize=(10, 6))
for num in model_numbers:
    model = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_"+str(num)+"_mask_900_LRELU_int_up_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    plt.plot(model["epoch"], model['loss'], label=f"Width: {name[i]}")
    i += 1
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation loss (MAE)")
plt.ylim([0.1, 0.2])
plt.xlim([0, 600])
# plt.savefig("Data/Data_for_ML/Model_history/Learning_width.pdf")
plt.show()

# Activation
model_names = ["LRELU", "RELU", "elu", "linear", "sigmoid", "tanh"]
name = ["Leaky ReLU", "ReLU", "ELU", "Linear", "Sigmoid", "Tanh"]
i = 0
plt.figure(figsize=(10, 6))
for num in model_names:
    model = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_55555_mask_900_"+str(num)+"_int_up_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
    plt.plot(model["epoch"], model['loss'], label=f"{name[i]}")
    i += 1

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation loss (MAE)")
plt.ylim([0.1, 0.4])
plt.xlim([0, 600])
# plt.savefig("Data/Data_for_ML/Model_history/Learning_activation.pdf")
plt.show()

modeldndz = pd.read_csv("Data/Data_for_ML/Model_history/run"
                        "-Ensemble_model_1_6x5_mask_1899dndz_LRELU_int_validation-tag-epoch_loss.csv",
                        delimiter=",", names=model_headers, skiprows=1)
plt.plot(modeldndz['epoch'], modeldndz['loss'], label="dndz trained")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation loss (MAE)")
plt.ylim([0.1, 0.4])
plt.xlim([0, 400])
plt.show()

