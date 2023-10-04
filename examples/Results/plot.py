import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_compare(plot_data,metric):
    _ = plt.figure()
    metric,value_name = metric
    for (directory,label) in plot_data:
        abspath = os.path.abspath(f'../{directory}')
        if not os.path.isdir(abspath): print("Not found"); continue
        logs = [torch.load(f"{abspath}/{seed}/{metric}.pt") for seed in os.listdir(abspath) if seed.isnumeric()]
        # Create DataFrame and aggreegate
        df = pd.DataFrame(logs)
        df = df.T.rename_axis('Epoch').reset_index()
        df = df.melt(id_vars=["Epoch"],
                     value_vars=list(df.columns[1:]),
                     var_name="seed",
                     value_name=value_name)

        sns.lineplot(x="Epoch",y=value_name,data=df,ci="sd",label=label,
                     err_kws={"alpha":0.15})
    plt.legend(framealpha=0.5)

############################### QUANTUM RBM ####################################
EXPERIMENT = "GibbsRBM"
METRICS = [('accuracy','Testing Accuracy'),
           ('score_log','Score'),
           ('err','Reconstruction Error'),
           ('p_log','Precision'),
           ('r_log','Recall')]
PLOT_DATA = [('1_RBM/bas/8x8/CD-5','CD-5'),('1_RBM/bas/8x8/CD-20','CD-20')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

################################ GIBBS RBM #####################################

EXPERIMENT = "GibbsRBM"
METRICS = [('accuracy','Testing Accuracy'),
           ('score_log','Score'),
           ('err','Reconstruction Error'),
           ('p_log','Precision'),
           ('r_log','Recall')]
PLOT_DATA = [('1_RBM/bas/8x8/CD-5','CD-5'),('1_RBM/bas/8x8/CD-20','CD-20')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')
