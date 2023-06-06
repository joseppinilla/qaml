import qaml
import copy
import dimod
import torch
import dwave.system
import dwave.embedding
import dwave.preprocessing

import numpy as np

from dwave.system import FixedEmbeddingComposite
from qaml.sampler.base import BinaryQuadraticModelNetworkSampler
from dwave.preprocessing import ScaleComposite, SpinReversalTransformComposite


class QuantumAnnealingNetworkSampler(BinaryQuadraticModelNetworkSampler):

    sample_kwargs = {
        # DWaveSampler
        "num_reads":10, "annealing_time":20.0, "label":"QAML",
        "answer_mode":"raw", "auto_scale":False,
        # ScaleComposite (Default values overwritten by get_device())
        "bias_range":[-5.0, 5.0], "quadratic_range":[-3.0, 3.0],
        # FixedEmbeddingComposite
        "chain_strength":1.6, "chain_break_fraction":False,
        "chain_break_method":dwave.embedding.chain_breaks.majority_vote,
        "return_embedding": True,
        # SpinReversalTransformComposite
        "num_spin_reversal_transforms":1}

    embedding = None

    def __init__(self, model, embedding=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False, **conf):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta=beta)

        self.auto_scale = auto_scale

        self.device = self.get_device(failover,retry_interval,**conf)
        if embedding is None:
            embedding = qaml.minor.clique_from_cache(model,self,mask)
            assert embedding, "Embedding not found"

        edgelist = self.to_networkx_graph().edges
        self.embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        child = DummySampler(self.device) if test else self.device

        self.child = SpinReversalTransformComposite(
                        FixedEmbeddingComposite(
                            ScaleComposite(child),
                            self.embedding,scale_aware=True))

    @classmethod
    def get_device(cls, failover=False, retry_interval=-1, **conf):
        device = dwave.system.DWaveSampler(failover,retry_interval,**conf)
        cls.sample_kwargs['bias_range'] = device.properties['h_range']
        cls.sample_kwargs['quadratic_range'] = device.properties['j_range']
        return device

    def to_networkx_graph(self):
        self._networkx_graph = self.device.to_networkx_graph()
        return self._networkx_graph


class BatchQuantumAnnealingNetworkSampler(QuantumAnnealingNetworkSampler):

    batch_embeddings = None

    def __init__(self, model, batch_embeddings=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False **conf):
        BinaryQuadraticModelNetworkSampler.__init__(self,model,beta=beta)

        self.auto_scale = auto_scale

        self.device = self.get_device(failover,retry_interval,**conf)

        if batch_embeddings is None:
            batch_embeddings = list(qaml.minor.harvest_cliques(model,self,mask))
            assert batch_embeddings, "Embedding not found"
        self.embedding = self.combine_embeddings(batch_embeddings)
        self.batch_embeddings = batch_embeddings

        child = DummySampler(self.device) if test else self.device
        self.child = SpinReversalTransformComposite(
                        FixedEmbeddingComposite(
                            ScaleComposite(child),
                            self.embedding,scale_aware=True))

    def combine_embeddings(self, batch_embeddings):
        combined_emb = {}
        offset = len(self.model)
        for i,emb in enumerate(batch_embeddings):
            emb_i = {x+(i*offset):chain for x,chain in emb.items()}
            combined_emb.update(emb_i)
        edgelist = self.networkx_graph.edges
        embedding = dwave.embedding.EmbeddedStructure(edgelist,combined_emb)
        return embedding

    def combine_bqms(self, fixed_vars):
        if len(fixed_vars) == 0:
            ising = self.to_ising()
            batch_bqms = [ising.copy() for _ in self.batch_embeddings]
        else:
            batch_bqms = [self.to_ising(fix).copy() for fix in fixed_vars]

        offset = len(self.model)
        vartype = self.model.vartype
        combined_bqm = dimod.BinaryQuadraticModel.empty(vartype)
        for i,bqm in enumerate(batch_bqms):
            labels = {x:x+(i*offset) for x in bqm.variables}
            relabeled = bqm.relabel_variables(labels,inplace=False)
            combined_bqm.update(relabeled.copy())
        return combined_bqm

    @property
    def batch_size(self):
        return len(self.batch_embeddings) if self.batch_embeddings else None

    def sample_bm(self, fixed_vars=[], **kwargs):
        scalar = None if self.auto_scale else self.beta.item()
        sample_kwargs = {**self.sample_kwargs,**kwargs}
        vartype = self.model.vartype
        if len(fixed_vars) > self.batch_size:
            raise RuntimeError("Input batch size larger than sampler size")
        bqm = self.combine_bqms(fixed_vars)

        response = self.child.sample(bqm,scalar=scalar,**sample_kwargs)
        response.resolve()
        sampleset = response.change_vartype(vartype)
        info = sampleset.info.copy()
        variables = sampleset.variables.copy()
        batch_size = len(fixed_vars) if fixed_vars else self.batch_size

        # (num_reads,VARS*batch_size)
        samples = sampleset.record.sample.copy()
        # (num_reads,VARS*batch_size)   ->   (num_reads,VARS)*batch_size
        split_samples = np.split(samples,batch_size,axis=1)

        samplesets = []
        # All samples belong to the same BQM. Concatenate and return.
        if len(fixed_vars) == 0:
            for i,split in enumerate(split_samples):
                split_set = dimod.SampleSet.from_samples(split,vartype,np.nan,info=info)
                split_set.relabel_variables({k:v for k,v in enumerate(variables)})
                samplesets.append(split_set)
            return dimod.concatenate(samplesets)

        # Each sampleset is for a different input. Fill in and return.
        else:
            for fixed,split in zip(fixed_vars,split_samples):
                split_set = dimod.SampleSet.from_samples(split,vartype,np.nan,info=info)
                split_set.relabel_variables({k:v for k,v in enumerate(variables)})
                fixed_set = dimod.SampleSet.from_samples(fixed,vartype,np.nan)
                samplesets.append(dimod.append_variables(split_set,fixed_set))
            return samplesets

BatchQASampler = BatchQuantumAnnealingNetworkSampler
