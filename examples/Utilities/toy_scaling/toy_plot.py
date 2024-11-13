import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = ['[-0.1,0.1]','[-0.5,0.5]','[-1.0,1.0]','[-2.0,2.0]','[-4.0,4.0]']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd']

def plot_distance(data,filename):
    plt.figure()
    ax = plt.gca()
    bplot = ax.boxplot(data,labels=labels,
                       patch_artist=True,showmeans=True,whis=1.0)
    plt.ylim(0.0,1.0)
    ax.set_xlabel('Uniform Weight Range')
    ax.set_ylabel(r'$d(\beta_{eff})$', rotation=0)
    ax.yaxis.grid(True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for line in bplot['medians']:
        line.set_linewidth(2)
        line.set_color('#7f7f7f')
    for line in bplot['means']:
        line.set_markeredgecolor('#7f7f7f')
        line.set_markerfacecolor('#7f7f7f')
    plt.savefig(filename)

def plot_beta(data,filename):
    plt.figure()
    ax = plt.gca()
    bplot = ax.boxplot(data,labels=labels,
                       patch_artist=True,showmeans=True,whis=1.0)
    plt.ylim(2.0,10.0)
    ax.set_xlabel('Uniform Weight Range')
    ax.set_ylabel(r'$\beta_{eff}$', rotation=0)
    ax.yaxis.grid(True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for line in bplot['medians']:
        line.set_linewidth(2)
        line.set_color('#7f7f7f')
    for line in bplot['means']:
        line.set_markeredgecolor('#7f7f7f')
        line.set_markerfacecolor('#7f7f7f')
    plt.savefig(filename)

def plot_alpha(data,filename):
    plt.figure()
    ax = plt.gca()
    for line,label,color in zip(data,labels,colors):
        ax.plot(line,label=label,color=color)
    ax.set_xlabel('Uniform Weight Range')
    ax.set_ylabel(r'$\alpha$', rotation=0)
    ax.yaxis.grid(True)
    plt.legend()
    plt.savefig(filename)

# auto_scale = False
df = pd.read_csv("./dist_noscale.csv",delim_whitespace=True).transpose()
figa = df.values.tolist()
plot_distance(figa,"distance_noscale.eps")

df = pd.read_csv("./beta_noscale.csv",delim_whitespace=True).transpose()
figb = df.values.tolist()
plot_beta(figb,"beta_noscale.eps")

# auto_scale = True
df = pd.read_csv("./dist_scale.csv",delim_whitespace=True).transpose()
figa = df.values.tolist()
plot_distance(figa,"distance_scale.eps")

df = pd.read_csv("./beta_scale.csv",delim_whitespace=True).transpose()
figb =df.values.tolist()
plot_beta(figb,"beta_scale.eps")

df = pd.read_csv("./scalar.csv",delim_whitespace=True).transpose()
figc = df.values.tolist()
plot_alpha(figc,"alpha_scale.eps")
