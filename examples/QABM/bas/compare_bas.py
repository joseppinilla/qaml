%matplotlib qt

import torch
import minorminer
import matplotlib.pyplot as plt


################################ FULL ##########################################
SEED = 42
BATCH_SIZE = 83
HIDDEN_SIZE = 16
TYPE = 'full'

directory = f".{TYPE}/{HIDDEN_SIZE}_batch{BATCH_SIZE}"
directory = directory.replace('.','')
embedding = torch.load(f'./{TYPE}/embedding.pt')
source = torch.load(f'./{TYPE}/source.pt')
target = torch.load(f'./{TYPE}/target.pt')

miner = minorminer.miner(source,target)
miner.quality_key(embedding)
# (0, [], [9, 68, 8, 12])
################################ HEUR ##########################################
SEED = 42
BATCH_SIZE = 84
HIDDEN_SIZE = 16
TYPE = 'heur'

directory = f".{TYPE}/{HIDDEN_SIZE}_batch{BATCH_SIZE}"
directory = directory.replace('.','')

embedding = torch.load(f'./{TYPE}/embedding.pt')
source = torch.load(f'./{TYPE}/source.pt')
target = torch.load(f'./{TYPE}/target.pt')

miner = minorminer.miner(source,target)
miner.quality_key(embedding)
# (0, [], [16, 1, 15, 2, 14, 11, 13, 11, 12, 13, 11, 17, 10, 12, 9, 8, 8, 5])
################################ LOAD ##########################################
score_log = torch.load(f'{directory}/{SEED}/score_log.pt')
p_log = torch.load(f'{directory}/{SEED}/p_log.pt')
r_log = torch.load(f'{directory}/{SEED}/r_log.pt')
batch_err = torch.load(f'{directory}/{SEED}/batch_err.pt')
err = torch.load(f'{directory}/{SEED}/err.pt')
accuracy = torch.load(f'{directory}/{SEED}/accuracy.pt')

b_log = torch.load(f'{directory}/{SEED}/b.pt')
c_log = torch.load(f'{directory}/{SEED}/c.pt')
W_log = torch.load(f'{directory}/{SEED}/W.pt')
vv_log = torch.load(f'{directory}/{SEED}/vv.pt')
hh_log = torch.load(f'{directory}/{SEED}/hh.pt')


chain_length = {k:len(chain) for k,chain in embedding.items()}


bias_data = {k:[] for k in embedding}
for b_set,c_set in zip(b_log,c_log):
    for i,b in enumerate(b_set):
        bias_data[i].append(b)
    for j,c in enumerate(c_set,start=64):
        bias_data[j].append(c)

index= []
data = []
length = []
for i, (key, val) in enumerate(bias_data.items()):
    length.append(chain_length[i])
    index.append(key)
    data.append(val)

sorted_length, sorted_data = zip(*sorted(zip(length, data)))

gradient_data = []
for data in sorted_data:
    gradient_bias = []
    for a,b in zip(data[:-1],data[1:]):
        gradient_bias.append(b-a)
    gradient_data.append(gradient_bias)

fig, ax = plt.subplots(1)
_ = ax.boxplot(gradient_data)
_ = ax.set_xticklabels(sorted_length)



fig, ax = plt.subplots()
ax.plot(score_log)
plt.ylabel("Score")
plt.xlabel("Epoch")

fig, ax = plt.subplots()
ax.plot(p_log)
plt.ylabel("Precision")
plt.xlabel("Epoch")

fig, ax = plt.subplots()
ax.plot(r_log)
plt.ylabel("Recall")
plt.xlabel("Epoch")

fig, ax = plt.subplots()
ax.plot(batch_err)
plt.ylabel("Batch Error")
plt.xlabel("Epoch")

fig, ax = plt.subplots()
ax.plot(err)
plt.ylabel("Epoch Error")
plt.xlabel("Epoch")

fig, ax = plt.subplots()
ax.plot(accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
