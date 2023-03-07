
# Required packages
import os
import qaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

solver_name = "Advantage_system4.1"

def plot_compare(plot_data,metric,filename,yscale='linear'):
    log_name, value_name = metric
    _ = plt.figure()
    for (directory,label) in plot_data:
        abspath = os.path.abspath(f'./{directory}/{solver_name}')
        if not os.path.isdir(abspath): continue
        logs = []
        for seed in os.listdir(abspath):
            if seed.isnumeric():
                try: logs.append(torch.load(f"{abspath}/{seed}/{log_name}.pt"))
                except: continue
        # Create DataFrame and aggreegate
        df = pd.DataFrame(logs)
        df = df.T.rename_axis('Epoch').reset_index()
        df = df.melt(id_vars=["Epoch"],
                     value_vars=list(df.columns[1:]),
                     var_name="seed",
                     value_name=value_name)

        line = sns.lineplot(x="Epoch",y=value_name,data=df,ci="sd",label=label,
                     err_kws={"alpha":0.15})
    plt.yscale(yscale)

    plt.legend(framealpha=0.5)
    plt.savefig(filename)


plot_data = [('vanilla','Complete'),
             # ('adachi','Adachi w/ 24 edges pruned'),
             # ('adaptive','Adapt w/ 52 edges pruned'),
             # ('priority','Prio w/ 52 edges pruned'),
             # ('repurpose','Rep w/ 40 edges pruned + 1 hidden node'),
             ('heuristic','Heuristic')]


filename = "complete_v_heuristic_accuracy.svg"
metric = ("accuracy","Testing Accuracy")
plot_compare(plot_data,metric,filename,'linear')
filename = "complete_v_heuristic_error.svg"
metric = ("err","Error")
plot_compare(plot_data,metric,filename,'log')
filename = "complete_v_heuristic_kl_div.svg"
metric = ("kl_div","KL Divergence")
plot_compare(plot_data,metric,filename,'log')
