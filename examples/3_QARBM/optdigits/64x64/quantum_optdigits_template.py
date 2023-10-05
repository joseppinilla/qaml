# %% v1.3
# Quantum RBM training on the OptDigits Dataset for reconstruction and classification
# This is an example on quantum annealing training of an RBM on the OptDigits
# dataset.
# Developed by: Jose Pinilla

# Required packages
import os
import qaml
import torch

################################# Hyperparameters ##############################
EPOCHS = 75
M,N = SHAPE = (8,8)
SUBCLASSES = [0,1,2,3,5,6,7,8]
BATCH_SIZE = 1024
TEST_READS = 20

# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

# Set up training mechanisms
method = 'vanilla'
solver_name = "Advantage_system6.2"
sampler_kwargs = {'auto_scale':True,'num_spin_reversal_transforms':4}

# Deterministic results
SEED = 3
_ = torch.manual_seed(SEED)

#################################### Input Data ################################
opt_train = qaml.datasets.OptDigits(root='../data/',train=True,download=True,
                                        transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_train,SUBCLASSES)
qaml.datasets._embed_labels(opt_train,encoding='one_hot',scale=255)
train_sampler = torch.utils.data.RandomSampler(opt_train,replacement=False)
train_loader = torch.utils.data.DataLoader(opt_train,BATCH_SIZE,sampler=train_sampler)


opt_test = qaml.datasets.OptDigits(root='../data/',train=False,download=True,
                                       transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_test,SUBCLASSES)
set_label,get_label = qaml.datasets._embed_labels(opt_test,encoding='one_hot',
                                                  scale=255,setter_getter=True)

################################# Model Definition #############################
VISIBLE_SIZE = M*N
HIDDEN_SIZE = 64

# Specify model with dimensions
rbm = qaml.nn.RBM(VISIBLE_SIZE,HIDDEN_SIZE,'SPIN')

# Set up optimizer
beta = 2.5
optimizer = torch.optim.SGD(rbm.parameters(),lr=learning_rate/beta,
                            weight_decay=weight_decay,momentum=momentum)

# Loss and autograd
CD = qaml.autograd.ContrastiveDivergence

# Set up training mechanisms
verbose = True
if method == 'vanilla':
    neg_sampler = qaml.sampler.QASampler(rbm,beta=beta,solver=solver_name,test=True)

elif method == 'adachi':
    qa_sampler = qaml.sampler.AdachiQASampler(rbm,solver=solver_name,beta=beta)
    _ = qaml.prune.adaptive_unstructured(rbm,'W',qa_sampler,verbose)

elif method == 'adaptive':
    qa_sampler = qaml.sampler.AdaptiveQASampler(rbm,solver=solver_name,beta=beta)
    _ = qaml.prune.adaptive_unstructured(rbm,'W',qa_sampler,verbose)

elif method == 'priority': #TODO: argument should be mask, not list
    priority = set_label(torch.zeros(1,*SHAPE),1).flatten()
    qa_sampler = qaml.sampler.AdaptiveQASampler(rbm,solver=solver_name,beta=beta)
    _ = qaml.prune.priority_unstructured(rbm,'W',qa_sampler,priority,verbose)

elif method == 'repurpose':
    qa_sampler = qaml.sampler.RepurposeQASampler(rbm,solver=solver_name,beta=beta)
    _ = qaml.prune.adaptive_unstructured(rbm,'W',qa_sampler,verbose)

elif method == 'heuristic':
    qa_sampler = qaml.sampler.QASampler(rbm,{},beta,solver=solver_name)
    qa_sampler.set_embedding(qaml.minor.miner_heuristic(rbm,qa_sampler,SEED))

pos_sampler =  qaml.sampler.GibbsNetworkSampler(rbm,BATCH_SIZE)

################################## Training Log ################################

directory = f"{method}/{solver_name}/{SEED}"
os.makedirs(f'{directory}',exist_ok=True)

torch.save(rbm.to_networkx_graph(),f"{directory}/graph.pt")
torch.save(dict(neg_sampler.embedding),f"{directory}/embedding.pt")
torch.save(rbm.state_dict().get('W_mask',{}),f"{directory}/mask.pt")

err_log = []
kl_div_log = []
scalar_log = []
accuracy_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy()]

################################## Model Training ##############################
rbm.train()
for t in range(1):
    print(f"Epoch {t}")
    epoch_error = torch.Tensor([0.])
    epoch_kl_div = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:

        input_data = img_batch.flatten(1)
        # Negative Phase
        vk, hk = neg_sampler(num_reads=BATCH_SIZE,**sampler_kwargs)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data,beta*neg_sampler.scalar)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply(neg_sampler,(v0,prob_h0), (vk,hk), *rbm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        print(rbm.W.grad)

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err
        epoch_kl_div += qaml.perf.free_energy_smooth_kl(rbm,v0,vk)

    # Parameter log
    scalar_log.append(qa_sampler.scalar)
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy())
    # Error Log
    kl_div_log.append(epoch_kl_div.item())
    print(f"KL Divergence = {epoch_kl_div.item()}")
    err_log.append(epoch_error.item())
    print(f"Reconstruction Error = {epoch_error.item()}")

    ############################## Classification ##########################
    pos_sampler.beta.data = torch.tensor(beta*qa_sampler.scalar)
    count = 0
    tests = len(opt_test)
    for test_data, test_label in opt_test:
        input_data =  test_data.flatten(1)
        mask = set_label(torch.ones_like(test_data),0).flatten()
        v_recon,h_recon = pos_sampler.reconstruct(input_data,mask=mask,k=5)
        label_pred = get_label(v_recon.view(-1,*SHAPE))
        if label_pred.argmax() == get_label(test_data).argmax():
            count+=1
    accuracy_log.append(count/tests)
    print(f"Testing accuracy: {count}/{tests} ({count/tests:.2f})")

# %%
########################### Save model and results #############################
torch.save(b_log,f"./{directory}/b.pt")
torch.save(c_log,f"./{directory}/c.pt")
torch.save(W_log,f"./{directory}/W.pt")
torch.save(err_log,f"./{directory}/err.pt")
torch.save(kl_div_log,f"./{directory}/kl_div.pt")
torch.save(scalar_log,f"./{directory}/scalar.pt")
torch.save(accuracy_log,f"./{directory}/accuracy.pt")
