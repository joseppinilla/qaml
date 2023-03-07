import qaml
import torch

BATCH_SIZE = 64
SUBCLASSES = [0,1,2,3,5,6,7,8]
opt_train = qaml.datasets.OptDigits(root='./data/',train=True,download=True,
                                        transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_train,SUBCLASSES)
qaml.datasets._embed_labels(opt_train,encoding='one_hot',scale=255)
train_sampler = torch.utils.data.RandomSampler(opt_train,replacement=False)
train_loader = torch.utils.data.DataLoader(opt_train,BATCH_SIZE,sampler=train_sampler)


opt_test = qaml.datasets.OptDigits(root='./data/',train=False,download=True,
                                       transform=qaml.datasets.ToSpinTensor())
qaml.datasets._subset_classes(opt_test,SUBCLASSES)
set_label,get_label = qaml.datasets._embed_labels(opt_test,encoding='one_hot',
                                                  scale=255,setter_getter=True)
test_sampler = torch.utils.data.RandomSampler(opt_test,False)
test_loader = torch.utils.data.DataLoader(opt_test,BATCH_SIZE,sampler=test_sampler)


for data,target in opt_train:
    break
type(data)
type(target)

test_loader.batch_size
for img_batch, labels_batch in train_loader:
    break
type(img_batch)
type(img_batch)



DATA_SIZE = 10
TRAIN,TEST = SPLIT = 28,12
phase_dataset = qaml.datasets.PhaseState(DATA_SIZE,labeled=True,
                                         transform=qaml.datasets.ToSpinTensor(),
                                         target_transform=qaml.datasets.ToSpinTensor())
train_dataset,test_dataset = torch.utils.data.random_split(phase_dataset,[*SPLIT])

train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,batch_size=7)
test_sampler = torch.utils.data.RandomSampler(test_dataset,replacement=False)
test_loader = torch.utils.data.DataLoader(test_dataset,sampler=test_sampler)

for data,target in phase_dataset:
    break
type(data)
type(target)


test_loader.batch_size
for img_batch, labels_batch in train_loader:
    break
type(img_batch)
type(img_batch)
target
