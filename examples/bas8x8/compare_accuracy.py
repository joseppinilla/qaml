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


# auto_scale = False
filename = "Quantum_noscale_beta10.svg"
plot_data = [('BAS88_lrWbeta10_bcW01noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.1 '),
             ('BAS88_lrWbeta10_bcW05noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.5 '),
             ('BAS88_lrWbeta10_bcW10noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W1.0 '),
             ('BAS88_lrWbeta10_bcW40noscale_200_batchAdv_wd00001_cs16',        r'Neg-phase @ W4.0$^\dagger$')]
plot_compare(plot_data,filename)

filename = "Quantum_noscale_beta20.svg"
plot_data = [('BAS88_lrWbeta20_bcW01noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.1'),
             ('BAS88_lrWbeta20_bcW05noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.5'),
             ('BAS88_lrWbeta20_bcW10noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W1.0'),
             ('BAS88_lrWbeta20_bcW40noscale_200_batchAdv_wd00001_cs16',        u"Neg-phase @ W4.0$^\u2021$")]
plot_compare(plot_data,filename)

filename = "Quantum_noscale_beta30.svg"
plot_data = [('BAS88_lrWbeta30_bcW01noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.1'),
             ('BAS88_lrWbeta30_bcW05noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.5'),
             ('BAS88_lrWbeta30_bcW10noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W1.0'),
             ('BAS88_lrWbeta30_bcW40noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W4.0')]
plot_compare(plot_data,filename)

filename = "Quantum_noscale_beta40.svg"
plot_data = [('BAS88_lrWbeta40_bcW01noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.1'),
             ('BAS88_lrWbeta40_bcW05noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W0.5'),
             ('BAS88_lrWbeta40_bcW10noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W1.0'),
             ('BAS88_lrWbeta40_bcW40noscale_200_batchAdv_wd00001_cs16',        'Neg-phase @ W4.0')]
plot_compare(plot_data,filename)

# auto_scale = True
filename = "Quantum_scale_beta10.svg"
plot_data = [('BAS88_lrWbeta10_bcW01scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.1'),
             ('BAS88_lrWbeta10_bcW05scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.5'),
             ('BAS88_lrWbeta10_bcW10scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W1.0'),
             ('BAS88_lrWbeta10_bcW40scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W4.0')]
plot_compare(plot_data,filename)

filename = "Quantum_scale_beta20.svg"
plot_data = [('BAS88_lrWbeta20_bcW01scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.1'),
             ('BAS88_lrWbeta20_bcW05scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.5'),
             ('BAS88_lrWbeta20_bcW10scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W1.0'),
             ('BAS88_lrWbeta20_bcW40scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W4.0')]
plot_compare(plot_data,filename)

filename = "Quantum_scale_beta30.svg"
plot_data = [('BAS88_lrWbeta30_bcW01scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.1'),
             ('BAS88_lrWbeta30_bcW05scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.5'),
             ('BAS88_lrWbeta30_bcW10scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W1.0'),
             ('BAS88_lrWbeta30_bcW40scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W4.0')]
plot_compare(plot_data,filename)

filename = "Quantum_scale_beta40.svg"
plot_data = [('BAS88_lrWbeta40_bcW01scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.1'),
             ('BAS88_lrWbeta40_bcW05scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W0.5'),
             ('BAS88_lrWbeta40_bcW10scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W1.0'),
             ('BAS88_lrWbeta40_bcW40scale_200_batchAdv_wd00001_cs16',        'Pos-phase @ W4.0')]
plot_compare(plot_data,filename)

# Classical

filename = "Classical_samples.pdf"
plot_data = [('BAS88_bcW01_classical5_beta10_200_wd00001',       'Classical bcW0.1'),
             ('BAS88_bcW05_classical5_beta10_200_wd00001',       'Classical bcW0.5'),
             ('BAS88_bcW10_classical5_beta10_200_wd00001',       'Classical bcW1.0'),
             ('BAS88_bcW40_classical5_beta10_200_wd00001',       'Classical bcW4.0')]
plot_compare(plot_data,filename)
