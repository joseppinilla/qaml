import copy
import dimod
import torch
import dwave.system
import dwave.embedding

import numpy as np

from qaml.minor import clique_from_cache
from qaml.sampler.base import BinaryQuadraticModelNetworkSampler

class QuantumAnnealingNetworkSampler(BinaryQuadraticModelNetworkSampler):

    sample_kwargs = {"annealing_time":20.0,"label":"QAML"}

    embed_kwargs = {"chain_strength":1.6}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}

    scalar = 1.0
    embedding = None
    auto_scale = True

    def __init__(self, model, embedding=None, beta=1.0, auto_scale=True,
                 failover=False, retry_interval=-1, mask=[], **conf):
        BinaryQuadraticModelSampler.__init__(self,model,beta=beta)

        self.device = dwave.system.DWaveSampler(failover,retry_interval,**conf)
        target = self.to_networkx_graph()
        self.auto_scale = auto_scale

        if embedding is None:
            embedding = clique_from_cache(model,self,mask)
            assert embedding, "Embedding not found"
        self.embedding = dwave.embedding.EmbeddedStructure(target.edges,embedding)

    @classmethod
    def get_device(cls, failover=False, retry_interval=-1, **conf):
        return dwave.system.DWaveSampler(failover,retry_interval,**conf)

    def set_embedding(self, embedding):
        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            target = self.networkx_graph
            embedding = dwave.embedding.EmbeddedStructure(target.edges,embedding)
        self.embedding = embedding

    def to_networkx_graph(self):
        self._networkx_graph = self.device.to_networkx_graph()
        return self._networkx_graph

    def embed_bm(self, ising, embedding=None, flip_variables=[], **embed_kwargs):

        bqm = ising.copy()
        for v in flip_variables:
            bqm.flip_variable(v)

        embedding = self.embedding if embedding is None else embedding
        target_bqm = embedding.embed_bqm(bqm,**embed_kwargs)
        ignoring = [e for u in embedding for e in embedding.chain_edges(u)]
        scale_kwargs = {'ignored_interactions':ignoring}

        if self.auto_scale:
            # Same function as sampler's auto_scale but retains scalar
            scale_kwargs.update({'bias_range':self.device.properties['h_range'],
                               'quadratic_range':self.device.properties['j_range']})
            self.scalar = target_bqm.normalize(**scale_kwargs)
        else:
            target_bqm.scale(1.0/float(self.beta),**scale_kwargs)
            self.scalar = 1.0

        return target_bqm

    def unembed_sampleset(self, response, embedding, bqm, **kwargs):

        sampleset = dwave.embedding.unembed_sampleset(response,embedding,bqm,
                                                      **kwargs)
        return sampleset

    def reconstruct(self, input_data, mask, num_reads=100,
                    embed_kwargs={}, unembed_kwargs={}, **kwargs):

        kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        fixed_vars = [{i:v.item() for i,v in enumerate(d) if mask[i]} for d in input_data]
        return self.sample_bm(fixed_vars,embed_kwargs,unembed_kwargs,**kwargs)

    def forward(self, input_data=[], num_reads=100,
                embed_kwargs={}, unembed_kwargs={}, **kwargs):

        kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        fixed_vars = [{i:v.item() for i,v in enumerate(d)} for d in input_data]
        return self.sample_bm(fixed_vars,embed_kwargs,unembed_kwargs,**kwargs)

class BatchQuantumAnnealingNetworkSampler(BinaryQuadraticModelNetworkSampler):

    sample_kwargs = {"annealing_time":20.0,"label":"QAML"}

    embed_kwargs = {"chain_strength":1.6}

    unembed_kwargs = {"chain_break_fraction":False,
                      "chain_break_method":dwave.embedding.chain_breaks.majority_vote}

    scalar = 1.0
    embedding = None
    auto_scale = True

    batch_mode = False
    batch_embeddings = None

    def __init__(self, model, embedding=None, beta=1.0, auto_scale=True,
                 chain_imbalance=0, failover=False, retry_interval=-1, mask=[],
                 batch_mode=False, **conf):
        BinaryQuadraticModelSampler.__init__(self,model,beta=beta)

        self.device = dwave.system.DWaveSampler(failover,retry_interval,**conf)
        target = self.to_networkx_graph()
        self.auto_scale = auto_scale

        if embedding is None:
            embedding = qaml.minor.clique_from_cache(model,self)
            assert embedding, "Embedding not found"
        self.embedding = dwave.embedding.EmbeddedStructure(target.edges,embedding)

        if batch_mode:
            self.batch_embeddings = list(qaml.minor.harvest_cliques(model,self,mask))
            self.batch_mode = len(self.batch_embeddings)

    @classmethod
    def get_device(cls, failover=False, retry_interval=-1, **conf):
        return dwave.system.DWaveSampler(failover,retry_interval,**conf)

    def set_embedding(self, embedding):
        if not isinstance(embedding,dwave.embedding.EmbeddedStructure):
            target = self.networkx_graph
            embedding = dwave.embedding.EmbeddedStructure(target.edges,embedding)
        self.embedding = embedding

    def to_networkx_graph(self):
        self._networkx_graph = self.device.to_networkx_graph()
        return self._networkx_graph

    def embed_bm(self, ising, embedding=None, flip_variables=[], **embed_kwargs):

        bqm = ising.copy()
        for v in flip_variables:
            bqm.flip_variable(v)

        embedding = self.embedding if embedding is None else embedding
        target_bqm = embedding.embed_bqm(bqm,**embed_kwargs)
        ignoring = [e for u in embedding for e in embedding.chain_edges(u)]
        scale_kwargs = {'ignored_interactions':ignoring}

        if self.auto_scale:
            # Same function as sampler's auto_scale but retains scalar
            scale_kwargs.update({'bias_range':self.device.properties['h_range'],
                               'quadratic_range':self.device.properties['j_range']})
            self.scalar = target_bqm.normalize(**scale_kwargs)
        else:
            target_bqm.scale(1.0/float(self.beta),**scale_kwargs)
            self.scalar = 1.0

        return target_bqm

    def sample_bm(self, fixed_vars=[], embed_kwargs={}, unembed_kwargs={}, **sample_kwargs):
        num_inputs = len(fixed_vars)
        batch_size = self.batch_mode
        edgelist = self._networkx_graph.edges
        if num_inputs == 0:
            print("No input")
            ising = self.to_ising()
            embedding = self.embedding
            num_variables = len(ising)
        elif num_inputs == 1:
            print("Fixed input")
            ising = self.to_ising(*fixed_vars)
            num_variables = len(ising)
            clamped_emb = {v:self.embedding[v] for v in ising.variables}
            embedding = dwave.embedding.EmbeddedStructure(edgelist,clamped_emb)
        elif num_inputs <= batch_size:
            print("Multiple inputs with batch embeddings")
            batch_embs = self.batch_embeddings
            batch_bqms = [self.to_ising(fix) for fix in fixed_vars]
            ising = dimod.BinaryQuadraticModel.empty(self.model.vartype)
            combined_emb = {}
            offset = self.model.V + self.model.H
            for i,(bqm,emb) in enumerate(zip(batch_bqms,batch_embs)):
                labels = {v:v+(i*offset) for v in bqm.variables}
                emb_i = {v+(i*offset):chain for v,chain in emb.items()}
                combined_emb.update(emb_i)
                relabeled = bqm.relabel_variables(labels,inplace=False)
                ising.update(relabeled)
            embedding = dwave.embedding.EmbeddedStructure(edgelist,combined_emb)
            num_variables = len(labels)
        else:
            raise ValueError(f'Input ({num_inputs}) != batch ({batch_size})')

        embed_kwargs = {**self.embed_kwargs,**embed_kwargs}
        sample_kwargs = {**self.sample_kwargs,**sample_kwargs}
        unembed_kwargs = {**self.unembed_kwargs,**unembed_kwargs}

        num_reads = sample_kwargs.pop('num_reads',100)
        auto_scale = sample_kwargs.pop('auto_scale',self.auto_scale)
        num_spinrevs = sample_kwargs.pop('num_spin_reversal_transforms',0)

        if num_spinrevs > 1:
            reads_per_transform = num_reads//num_spinrevs
            iter_num_reads = [reads_per_transform]*(num_spinrevs-1)
            iter_num_reads += [reads_per_transform+(num_reads%num_spinrevs)]
        else:
            iter_num_reads = [num_reads]

        transform = []
        responses = []

        for num_reads in iter_num_reads:

            # Don't flip if num_spin_reversal_transforms is 0
            if num_spinrevs > 0:
                transform = [v for v in ising.variables if np.random.rand() > .5]

            target_bqm = self.embed_bm(ising,embedding,flip_variables=transform,**embed_kwargs)
            target_response = self.device.sample(target_bqm,auto_scale=False,
                                          num_reads=num_reads,answer_mode='raw',
                                          num_spin_reversal_transforms=0,
                                          **sample_kwargs)
            target_response.resolve()
            target_response.change_vartype(self.model.vartype,inplace=True)

            flipped_response = self.unembed_sampleset(target_response,
                                                      embedding,ising,
                                                      **unembed_kwargs)
            tf_idxs = [flipped_response.variables.index(v) for v in transform]
            if self.model.vartype is dimod.BINARY:
                flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]
            elif self.model.vartype is dimod.SPIN:
                flipped_response.record.sample[:, tf_idxs] = -flipped_response.record.sample[:, tf_idxs]
            responses.append(flipped_response)

        sampleset = dimod.sampleset.concatenate(responses)

        if num_inputs == 0:
            return sampleset # (num_reads,V+H)
        elif num_inputs == 1:
            fixed_samples, fixed_labels = dimod.as_samples(fixed_vars) # (1,FIXED)
            repeated = np.repeat(fixed_samples,num_reads,axis=0) # (num_reads,FIXED)
            return np.hstack((repeated,samples))


        sampleset = dimod.sampleset.concatenate(responses)
        samples = sampleset.record.sample.copy() # (num_reads,variables)
        variables = sampleset.variables[:num_variables]

        if num_inputs == 0:
            self.sampleset = samples # (num_reads,V+H)
        elif num_inputs == 1:
            fixed, _ = dimod.as_samples(fixed_vars)
            repeated = np.repeat(fixed,num_reads,0) # (1,FIXED) -> (num_reads,FIXED)
            self.sampleset = np.hstack((repeated,samples))
        elif num_inputs <= batch_size:
            split_samples = np.split(samples,len(fixed_vars),axis=1) # [(num_reads,VARS)*BATCH_SIZE]
            mean_samples = np.mean(split_samples,axis=1) #(BATCH_SIZE,VARS)
            # TODO: Only return mean if return_prob?

            concatenate = copy.deepcopy(fixed_vars)
            for i,(fixed_batch,sample_batch) in enumerate(zip(fixed_vars,mean_samples)):
                concatenate[i].update({int(k):v for k,v in zip(variables,sample_batch)})

            # sort labels
            samples,variables = dimod.as_samples(concatenate)
            reindex, new_variables = zip(*sorted(enumerate(variables),
                                                     key=lambda tup: tup[1]))
            if new_variables != variables:
                    # avoid the copy if possible
                    samples = samples[:, reindex]
                    variables = new_variables
            self.sampleset = samples

        sampletensor = torch.tensor(self.sampleset,dtype=torch.float32)
        return sampletensor.split([self.model.V,self.model.H],1)

    def unembed_sampleset(self, response, embedding, bqm, **kwargs):

        sampleset = dwave.embedding.unembed_sampleset(response,embedding,bqm,
                                                      **kwargs)
        return sampleset

    def reconstruct(self, input_data, mask, num_reads=100,
                    embed_kwargs={}, unembed_kwargs={}, **kwargs):

        kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        fixed_vars = [{i:v.item() for i,v in enumerate(d) if mask[i]} for d in input_data]
        return self.sample_bm(fixed_vars,embed_kwargs,unembed_kwargs,**kwargs)

    def forward(self, input_data=[], num_reads=100,
                embed_kwargs={}, unembed_kwargs={}, **kwargs):

        kwargs = {**self.sample_kwargs,**kwargs,'num_reads':num_reads}
        fixed_vars = [{i:v.item() for i,v in enumerate(d)} for d in input_data]
        return self.sample_bm(fixed_vars,embed_kwargs,unembed_kwargs,**kwargs)

QASampler = QuantumAnnealingNetworkSampler
