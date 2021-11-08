%matplotlib qt
# Required packages
import os
import qaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filename = "Quantum_beta.pdf"
plot_data = [('BAS88_beta10_05noscale_200_batchAdv_wd0001',             'Quantum no_scale @ $\\beta_{eff}$=1.0'),
             ('BAS88_beta25_05noscale_200_batchAdv_wd0001',             'Quantum no_scale @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta40_05noscale_200_batchAdv_wd0001',             'Quantum no_scale @ $\\beta_{eff}$=4.0')]

filename = "Quantum_samples.pdf"
plot_data = [('BAS88_beta25_01scale_200_2000Adv_wd0001',        'Quantum scale batch @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta25_05scale_200_batchAdv_wd0001',       'Quantum scale 2000 @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_beta25_200_wd0001',             'Quantum scale 2000 @ $\\beta_{eff}$=2.5')]

for (directory,label) in plot_data:
    abspath = os.path.abspath(f'./bas8x8/{directory}')
    if not os.path.isdir(abspath): continue
    logs = [torch.load(f"{abspath}/{seed}/accuracy.pt") for seed in os.listdir(abspath) if seed.isnumeric()]
    # Create DataFrame and aggreegate
    df = pd.DataFrame(logs)
    df = df.T.rename_axis('Epoch').reset_index()
    df = df.melt(id_vars=["Epoch"],
                 value_vars=list(df.columns[1:]),
                 var_name="seed",
                 value_name="Testing Accuracy")

    sns.lineplot(x="Epoch", y="Testing Accuracy", data=df, ci="sd", label=label)
plt.legend(loc="lower right")
plt.savefig(filename)

####### PLOT ALL
filter = 'scale'
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('tab20',20).colors))
for path in os.listdir('./bas8x8/'):
    abspath = f"./bas8x8/{path}"
    if not os.path.isdir(abspath): continue
    if filter not in path: continue
    logs = [torch.load(abspath+f'/{seed}/accuracy.pt') for seed in os.listdir(abspath) if seed.isnumeric()]
    # Create DataFrame and aggreegate
    df = pd.DataFrame(logs)
    df = df.T.rename_axis('Epoch').reset_index()
    df = df.melt(id_vars=["Epoch"],
                 value_vars=list(df.columns[1:]),
                 var_name="seed",
                 value_name="Testing Accuracy")

    sns.lineplot(x="Epoch", y="Testing Accuracy", data=df, ci="sd", label=path, lw=5)



####### PLOT INDIVIDUAL AND SAVE
for path in os.listdir('./bas8x8/'):
    abspath = f"./bas8x8/{path}"
    if not os.path.isdir(abspath): continue
    if not "BAS88" in abspath: continue
    fig, ax = plt.subplots()
    logs = [torch.load(abspath+f'/{seed}/accuracy.pt') for seed in os.listdir(abspath) if seed.isnumeric()]
    # Create DataFrame and aggreegate
    df = pd.DataFrame(logs)
    df = df.T.rename_axis('Epoch').reset_index()
    df = df.melt(id_vars=["Epoch"],
                 value_vars=list(df.columns[1:]),
                 var_name="seed",
                 value_name="Testing Accuracy")

    sns.lineplot(x="Epoch", y="Testing Accuracy", data=df, ci="sd", label=path, lw=5)
    plt.savefig(abspath+'/accuracy.png')
