import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_compare(plot_data,metric,epochs=-1,ylim=None,**fig_kwargs):
    _ = plt.figure(**fig_kwargs)
    metric,value_name = metric
    for (directory,label) in plot_data:
        abspath = os.path.abspath(f'../{directory}')
        if not os.path.isdir(abspath): print(f"Not found: {abspath}"); continue
        logs = []
        for seed in os.listdir(abspath):
            filepath = f"{abspath}/{seed}/{metric}.pt"
            if seed.isnumeric():
                if os.path.exists(filepath):
                    logs.append(torch.load(filepath)[:epochs])
        # Create DataFrame and aggreegate
        df = pd.DataFrame(logs)
        df = df.T.rename_axis('Epoch').reset_index()
        df = df.melt(id_vars=["Epoch"],
                     value_vars=list(df.columns[1:]),
                     var_name="seed",
                     value_name=value_name)

        lp = sns.lineplot(x="Epoch",y=value_name,data=df,errorbar="se",label=label,
                     err_kws={"alpha":0.15})
        if ylim:
            lp.set(ylim=ylim)
    plt.legend(framealpha=0.5)

################################# BM BAS #################################
EXPERIMENT = "BM_BAS_SA"
SA_SUBDIR = "2_BM/bas"
QA_SUBDIR = "4_QABM/bas"
METRICS = [#("err_log","Reconstruction Error"),
            ("p_log","Precision"),
            ("r_log","Recall"),
            ("score_log","Score"),]

# bm{V}_{H}-{TRAIN_READS}_{NUM_SWEEPS}
PLOT_DATA = [(f'{SA_SUBDIR}/bm36_16-10_1000/full/','SA'),
             (f'{SA_SUBDIR}/bm36_16-10_1000/0.8/','SA 0.8'),
             (f'{SA_SUBDIR}/bm36_16-10_1000/0.7/','SA 0.7'),]
             # (f'{QA_SUBDIR}/bm36_16-100/full/','QA'),
             # (f'{QA_SUBDIR}/bm36_16-100/0.8/','QA 0.8'),
             # (f'{QA_SUBDIR}/bm36_16-100/0.7/','QA 0.7')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label),50,figsize=(6,6))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

################################ BM BAS #################################
EXPERIMENT = "BM_BAS_SYS_HEUR"
SA_SUBDIR = "2_BM/bas"
QA_SUBDIR = "4_QABM/bas"
METRICS = [#("err_log","Reconstruction Error"),
            ("p_log","Precision"),
            ("r_log","Recall"),
            ("score_log","Score"),]

# bm{V}_{H}-{TRAIN_READS}_{NUM_SWEEPS}
PLOT_DATA = [(f'{SA_SUBDIR}/bm36_16-10_1000/full/','SA'),
             (f'{QA_SUBDIR}/bm36_16-100/full/','QA sys'),
             (f'{QA_SUBDIR}/bm36_16-100/heur/','QA heur'),]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label),50,figsize=(6,6))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

################################# BM OptDigits #################################
EXPERIMENT = "BM_OptDigits"
SUBDIR = "4_QABM/optdigits"
METRICS = [('accuracy','Testing Accuracy'),
            ("err","Reconstruction Error"),]

PLOT_DATA = [(f'{SUBDIR}/heur/16_batch84','Heuristic'),
             (f'{SUBDIR}/full/16_batch84','Systematic')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

####################### BM Complete vs Heuristic BAS ###########################
EXPERIMENT = "BM_Complete_v_Heuristic"
SUBDIR = "4_QABM/bas"
METRICS = [('accuracy','Testing Accuracy'),
            ("err","Reconstruction Error"),]

PLOT_DATA = [(f'{SUBDIR}/full/16_batch83','Complete'),
             (f'{SUBDIR}/heur/16_batch84','Heuristic')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')


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


################################## Adaptive ####################################
EXPERIMENT = "RBM_Adaptive"
SUBDIR = "3_QARBM/optdigits/64x64"
METRICS = [('accuracy','Testing Accuracy'),('kl_div','KL Divergence'),('err','Error')]
PLOT_DATA = [(f'{SUBDIR}/vanilla/Gibbs',   'Biclique'),
             (f'{SUBDIR}/adachi/Gibbs',    '24 edges pruned'),
             (f'{SUBDIR}/adaptive/Gibbs',  '52 edges pruned'),]
             # (f'{SUBDIR}/repurpose/Gibbs', '104 edges pruned + 1 hidden node')]

if not os.path.exists(f"./{EXPERIMENT}/"): os.makedirs(f"./{EXPERIMENT}/")
for metric,label in METRICS:
    plot_compare(PLOT_DATA,(metric,label))
    plt.savefig(f'./{EXPERIMENT}/{metric}.svg')

######################### RBM Complete vs Heuristic ############################
EXPERIMENT = "RBM_Systematic_v_Heuristic"
SUBDIR = "3_QARBM/optdigits/64x64"
METRICS = [('accuracy','Testing Accuracy'),
            ("err","Reconstruction Error"),]

PLOT_DATA = [(f'{SUBDIR}/vanilla','Systematic'),
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
