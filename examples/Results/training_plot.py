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
        if not os.path.isdir(abspath): print(f"Not found: {abspath}"); continue
        logs = []
        for seed in os.listdir(abspath):
            filepath = f"{abspath}/{seed}/{metric}.pt"
            if seed.isnumeric():
                if os.path.exists(filepath):
                    logs.append(torch.load(filepath))
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


############################### Exact v Quantum ################################
EXPERIMENT = "BM_Exact_v_Quantum"
METRICS = [('score_log','Score'),
           ('epoch_err_log','Reconstruction Error'),
           ('p_log','Precision'),
           ('r_log','Recall')]
PLOT_DATA = [(f'2_BM/phase',   'Exact'),
             (f'4_QABM/phase', 'Quantum')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

################################## Adaptive ####################################
EXPERIMENT = "QARBM_Adaptive"
SUBDIR = "3_QARBM/optdigits/64x64"
METRICS = [('accuracy','Testing Accuracy')]
PLOT_DATA = [(f'{SUBDIR}/adachi',   'Adachi w/ 24 edges pruned'),
             (f'{SUBDIR}/adaptive', 'Adapt w/ 52 edges pruned'),
             (f'{SUBDIR}/priority', 'Prio w/ 52 edges pruned'),
             (f'{SUBDIR}/repurpose','Rep w/ 40 edges pruned + 1 hidden node')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

########################### Complete vs Heuristic ##############################
EXPERIMENT = "Complete_v_Heuristic"
SUBDIR = "3_QARBM/optdigits/64x64"
METRICS = [('accuracy','Testing Accuracy'),
            ("err","Reconstruction Error"),]

PLOT_DATA = [(f'{SUBDIR}/vanilla','Complete'),
             (f'{SUBDIR}/heuristic','Heuristic')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

######################## Negative-phase Temp Scaling ###########################
BETAS = [10,20,30,40]
for beta in BETAS:
    EXPERIMENT = f"QARBM-BAS88-negphase"
    METRICS = [('accuracy','Testing Accuracy')]
    SUBDIR = '3_QARBM/bas/8x8/'
    PLOT_DATA = [(f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW01noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.1 '),
                 (f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW05noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.5 '),
                 (f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW10noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W1.0 '),
                 (f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW40noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W4.0 ')]

    if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
    for metric,label in METRICS:
        plot_compare(PLOT_DATA,(metric,label))
        plt.savefig(f'./{EXPERIMENT}/beta_{beta}_{metric}.svg')

######################## Positive-phase Temp Scaling ###########################
BETAS = [10,20,30,40]
for beta in BETAS:
    EXPERIMENT = f"QARBM-BAS88-posphase"
    METRICS = [('accuracy','Testing Accuracy')]
    SUBDIR = '3_QARBM/bas/8x8/'
    PLOT_DATA = [(f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW01scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.1 '),
                 (f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW05scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.5 '),
                 (f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW10scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W1.0 '),
                 (f'{SUBDIR}/BAS88_lrWbeta{beta}_bcW40scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W4.0 ')]

    if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
    for metric,label in METRICS:
        plot_compare(PLOT_DATA,(metric,label))
        plt.savefig(f'./{EXPERIMENT}/beta_{beta}_{metric}.svg')

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
