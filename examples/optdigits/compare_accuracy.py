%matplotlib qt
# Required packages
import os
import qaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_compare(plot_data,filename):
    _ = plt.figure()
    for (directory,label) in plot_data:
        abspath = os.path.abspath(f'./{directory}')
        if not os.path.isdir(abspath): continue
        logs = [torch.load(f"{abspath}/{seed}/accuracy.pt") for seed in os.listdir(abspath) if seed.isnumeric()]
        # Create DataFrame and aggreegate
        df = pd.DataFrame(logs)
        df = df.T.rename_axis('Epoch').reset_index()
        df = df.melt(id_vars=["Epoch"],
                     value_vars=list(df.columns[1:]),
                     var_name="seed",
                     value_name="Testing Accuracy")

        sns.lineplot(x="Epoch",y="Testing Accuracy",data=df,ci="sd",label=label,
                     err_kws={"alpha":0.15})

    plt.legend(framealpha=0.5)
    plt.savefig(filename)

filename = "Quantum_embedding_64_beta25.svg"
plot_data = [('Adv64_beta25','Complete'),
             ('Adachi64_beta25','Adachi w/ 24 edges pruned'),
             ('Adapt64_beta25','Adapt w/ 52 edges pruned'),
             ('Prio64_beta25','Prio w/ 52 edges pruned'),
             ('Rep64_beta25','Rep w/ 40 edges pruned + 1 hidden node')]
plot_compare(plot_data,filename)
