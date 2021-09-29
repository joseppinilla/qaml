# Required packages
import os
import qaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""                              QUANTUM VS CLASSICAL                        """

filename = "Quantum_v_Gibbs.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv',             'Quantum @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_200',                   'Gibbs-5')]

filename = "Quantum_v_Gibbs_A.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adachi',      'Quantum Pruned A @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_200_Adachi',            'Gibbs-5 Pruned A')]

filename = "Quantum_v_Gibbs_B.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adapt',       'Quantum Pruned B @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_200_Adapt',             'Gibbs-5 Pruned B')]

filename = "Quantum_v_Gibbs_C.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_AdaptPrio',   'Quantum Pruned C @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_200_AdaptPrio',         'Gibbs-5 Pruned C')]

"""                     QUANTUM VS CLASSICAL  @ BETA=2.5                      """

filename = "Quantum_v_Gibbs_betas.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_wd001',             'Quantum @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta40_scale_200_Adv_wd001',             'Quantum @ $\\beta_{eff}$=4.0'),
             ('BAS88_classical5_beta40_200',            'Gibbs-5 @ $\\beta_{eff}$=4.0'),
             ('BAS88_classical5_beta25_200',            'Gibbs-5 @ $\\beta_{eff}$=2.5')]

filename = "Quantum_v_Gibbs_A_beta25.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adachi',      'Quantum Pruned A @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_beta25_200_Adachi',            'Gibbs-5 Pruned A @ $\\beta_{eff}$=2.5')]

filename = "Quantum_v_Gibbs_B_beta25.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adapt',       'Quantum Pruned B @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_beta25_200_Adapt',             'Gibbs-5 Pruned B @ $\\beta_{eff}$=2.5')]

filename = "Quantum_v_Gibbs_C_beta25.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_AdaptPrio',   'Quantum Pruned C @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_beta25_200_AdaptPrio',         'Gibbs-5 Pruned C @ $\\beta_{eff}$=2.5')]

"""                     QUANTUM VS CLASSICAL W/ TRAIN BETA                   """

filename = "Quantum_v_Gibbs_adabeta25.pdf"
plot_data = [('BAS88_adabeta25_scale_200_Adv',             'Quantum w/ Adaptive $\\beta_{eff}$'),
             ('BAS88_classical5_200',                    'Gibbs-5')]

filename = "Quantum_v_Gibbs_A_beta25.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adachi',      'Quantum Pruned A @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_beta25_200_Adachi',            'Gibbs-5 Pruned A @ $\\beta_{eff}$=2.5')]

filename = "Quantum_v_Gibbs_B_beta25.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adapt',       'Quantum Pruned B @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_beta25_200_Adapt',             'Gibbs-5 Pruned B @ $\\beta_{eff}$=2.5')]

filename = "Quantum_v_Gibbs_C_beta25.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_AdaptPrio',   'Quantum Pruned C @ $\\beta_{eff}$=2.5'),
             ('BAS88_classical5_beta25_200_AdaptPrio',         'Gibbs-5 Pruned C @ $\\beta_{eff}$=2.5')]

"""                                    CLASSICAL                             """

filename = "Gibbs-k.pdf"
plot_data = [('BAS88_classical1_200',                   'Gibbs-1'),
             ('BAS88_classical5_200',                   'Gibbs-5'),
             ('BAS88_classical10_200',                  'Gibbs-10'),
             ('BAS88_classical50_200',                  'Gibbs-50')]

filename = "Gibbs_Pruned.pdf"
plot_data = [('BAS88_classical5_200',                   'Gibbs-5'),
            ('BAS88_classical5_200_Adachi',             'Gibbs-5 Pruned A'),
            ('BAS88_classical5_200_Adapt',              'Gibbs-5 Pruned B'),
            ('BAS88_classical5_200_AdaptPrio',          'Gibbs-5 Pruned C')]

"""                                     QUANTUM                              """

filename = "Quantum_Pruned_ABC.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adachi',      'Quantum Pruned A @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta25_scale_200_Adv_Adapt',       'Quantum Pruned B @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta25_scale_200_Adv_AdaptPrio',   'Quantum Pruned C @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta25_scale_200_Adv',             'Quantum @ $\\beta_{eff}$=2.5')]

filename = "Quantum_Pruned_AC.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv',             'Quantum @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta25_scale_200_Adv_Adachi',      'Quantum Pruned A @ $\\beta_{eff}$=2.5'),
             ('BAS88_beta25_scale_200_Adv_AdaptPrio',   'Quantum Pruned C @ $\\beta_{eff}$=2.5')]

"""                        QUANTUM w/ TRAIN BETA (/beta^2)                   """

filename = "Quantum_beta.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv',             'Quantum @ $\\beta_{eff}$=2.5'),
             ('BAS88_adabeta25_scale_200_Adv',            'Quantum w/ Adaptive $\\beta_{eff}$')]

filename = "Quantum_Adachi_beta.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adachi',      'Quantum Pruned A @ $\\beta_{eff}$=2.5'),
             ('BAS88_adabeta25_scale_200_Adv_Adachi',     'Quantum Pruned A w/ Adaptive $\\beta_{eff}$')]

filename = "Quantum_Adapt_beta.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_Adapt',      'Quantum Pruned B @ $\\beta_{eff}$=2.5'),
             ('BAS88_adabeta25_scale_200_Adv_Adapt',     'Quantum Pruned B w/ Adaptive $\\beta_{eff}$')]

filename = "Quantum_AdaptPrio_beta.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_AdaptPrio',   'Quantum Pruned C @ $\\beta_{eff}$=2.5'),
             ('BAS88_adabeta25_scale_200_Adv_AdaptPrio',  'Quantum Pruned C w/ Adaptive $\\beta_{eff}$'),
             ('BAS88_classical5_beta25_200_AdaptPrio',         'Gibbs-5 Pruned C @ $\\beta_{eff}$=2.5')]

"""                            QUANTUM w/ TRAIN BETA                         """

filename = "Quantum_beta.pdf"
plot_data = [('BAS88_beta25_scale_200_Adv_wd001',             'Quantum @ $\\beta_{eff}$=2.5'),
             ('BAS88_ada_beta1_scale_200_Adv',           'Quantum w/ Adaptive $\\beta_{eff}$'),
             ('BAS88_adabeta25_scale_200_Adv_wd00001',            'Quantum w/ Adaptive $\\beta_{eff}/\\beta$')]

%matplotlib qt
for (directory,label) in plot_data:
    abspath = os.path.abspath(f'./bas8x8/{directory}')
    logs = [torch.load(f"{abspath}/{seed}/accuracy.pt") for seed in os.listdir(abspath)]

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
