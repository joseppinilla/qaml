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
from dwave.preprocessing import ScaleComposite


class QuantumAnnealingNetworkSampler(BinaryQuadraticModelNetworkSampler):

    sample_kwargs = {
        # DWaveSampler
        "num_reads":10, "annealing_time":20.0, "label":"QAML",
        "answer_mode":"raw", "auto_scale":True,
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
        # If embedding not provided. Allows empty embedding {}
        if embedding is None:
            # Try biclique if RBM
            if 'Restricted' in str(model):
                embedding = qaml.minor.biclique_from_cache(model,self,mask)
            # Try clique otherwise and if biclique fails
            if not embedding:
                embedding = qaml.minor.clique_from_cache(model,self,mask)
            assert embedding, "Embedding not found"

        edgelist = self.to_networkx_graph().edges
        self.embedding = dwave.embedding.EmbeddedStructure(edgelist,embedding)

        if test:
            print('TEST MODE ON')
            child = DummySampler(self.device)
        else:
            child = self.device
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

QASampler = QuantumAnnealingNetworkSampler

class BatchQuantumAnnealingNetworkSampler(QuantumAnnealingNetworkSampler):

    batch_embeddings = None

    def __init__(self, model, batch_embeddings=None, mask=None, auto_scale=True,
                 beta=1.0, failover=False, retry_interval=-1, test=False, **conf):
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

        batch_size = len(fixed_vars) if fixed_vars else self.batch_size

        # (num_reads,VARS*batch_size)
        samples = sampleset.record.sample.copy()
        # (num_reads,VARS*batch_size)   ->   (num_reads,VARS)*batch_size
        split_samples = np.split(samples,batch_size,axis=1)

        info = sampleset.info.copy()
        variables = sampleset.variables.copy()
        # All samples belong to the same BQM. Concatenate and return.
        samplesets = []
        if len(fixed_vars) == 0:
            for i,split in enumerate(split_samples):
                split_set = dimod.SampleSet.from_samples(split,vartype,np.nan,info=info)
                split_set.relabel_variables({k:v for k,v in enumerate(variables)})
                samplesets.append(split_set)
            return dimod.concatenate(samplesets)

        # Or each sampleset is for a different input. Fill in and return.
        for fixed,split in zip(fixed_vars,split_samples):
            split_set = dimod.SampleSet.from_samples(split,vartype,np.nan,info=info)
            split_set.relabel_variables({k:v for k,v in enumerate(variables)})
            fixed_set = dimod.SampleSet.from_samples(fixed,vartype,np.nan)
            samplesets.append(dimod.append_variables(split_set,fixed_set))
        return samplesets

BatchQASampler = BatchQuantumAnnealingNetworkSampler

def DummySampler(device):
    target = device.to_networkx_graph()
    nodelist = target.nodes
    edgelist = target.edges
    return dimod.StructureComposite(dimod.RandomSampler(),nodelist,edgelist)


import typing

class SpinReversalTransformComposite(dimod.core.Sampler, dimod.core.Composite):
    """NOTE: MODIFIED TO PRESERVE SAMPLESET INFO

    Composite for applying spin reversal transform preprocessing.

    Spin reversal transforms (or "gauge transformations") are applied
    by randomly flipping the spin of variables in the Ising problem. After
    sampling the transformed Ising problem, the same bits are flipped in the
    resulting sample [#km]_.

    Args:
        sampler: A `dimod` sampler object.

        seed: As passed to :class:`numpy.random.default_rng`.

    Examples:
        This example composes a dimod ExactSolver sampler with spin transforms then
        uses it to sample an Ising problem.

        >>> from dimod import ExactSolver
        >>> from dwave.preprocessing.composites import SpinReversalTransformComposite
        >>> base_sampler = ExactSolver()
        >>> composed_sampler = SpinReversalTransformComposite(base_sampler)
        ... # Sample an Ising problem
        >>> response = composed_sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> response.first.sample
        {'a': -1, 'b': -1}

    References
    ----------
    .. [#km] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

    """
    _children: typing.List[dimod.core.Sampler]
    _parameters: typing.Dict[str, typing.Sequence[str]]
    _properties: typing.Dict[str, typing.Any]

    def __init__(self, child: dimod.core.Sampler, *, seed=None):
        self._child = child
        self.rng = np.random.default_rng(seed)

    @property
    def children(self) -> typing.List[dimod.core.Sampler]:
        try:
            return self._children
        except AttributeError:
            pass

        self._children = children = [self._child]
        return children

    @property
    def parameters(self) -> typing.Dict[str, typing.Sequence[str]]:
        try:
            return self._parameters
        except AttributeError:
            pass

        self._parameters = parameters = dict(spin_reversal_variables=tuple())
        parameters.update(self._child.parameters)
        return parameters

    @property
    def properties(self) -> typing.Dict[str, typing.Any]:
        try:
            return self._properties
        except AttributeError:
            pass

        self._properties = dict(child_properties=self._child.properties)
        return self._properties

    class _SampleSets:
        def __init__(self, samplesets: typing.List[dimod.SampleSet]):
            self.samplesets = samplesets

        def done(self) -> bool:
            return all(ss.done() for ss in self.samplesets)

    @dimod.decorators.nonblocking_sample_method
    def sample(self, bqm: dimod.BinaryQuadraticModel, *,
               num_spin_reversal_transforms: int = 1,
               **kwargs,
               ):
        """Sample from the binary quadratic model.

        Args:
            bqm: Binary quadratic model to be sampled from.

            num_spin_reversal_transforms:
                Number of spin reversal transform runs.
                A value of ``0`` will not transform the problem.
                If you specify a nonzero value, each spin reversal transform
                will result in an independent run of the child sampler.

        Returns:
            A sample set. Note that for a sampler that returns ``num_reads`` samples,
            the sample set will contain ``num_reads*num_spin_reversal_transforms`` samples.

        Examples:
            This example runs 100 spin reversals applied to one variable of a QUBO problem.

            >>> from dimod import ExactSolver
            >>> from dwave.preprocessing.composites import SpinReversalTransformComposite
            >>> base_sampler = ExactSolver()
            >>> composed_sampler = SpinReversalTransformComposite(base_sampler)
            ...
            >>> Q = {('a', 'a'): -1, ('b', 'b'): -1, ('a', 'b'): 2}
            >>> response = composed_sampler.sample_qubo(Q,
            ...               num_spin_reversal_transforms=100)
            >>> len(response)
            400
        """
        sampler = self._child
        # No SRTs, so just pass the problem through
        if not num_spin_reversal_transforms or not bqm.num_variables:
            sampleset = sampler.sample(bqm, **kwargs)
            # yield twice because we're using the @nonblocking_sample_method
            yield sampleset  # this one signals done()-ness
            yield sampleset  # this is the one actually used by the user
            return

        # we'll be modifying the BQM, so make a copy
        bqm = bqm.copy()

        # We maintain the Leap behavior that num_spin_reversal_transforms == 1
        # corresponds to a single problem with randomly flipped variables.

        # Get the SRT matrix
        SRT = self.rng.random((num_spin_reversal_transforms, bqm.num_variables)) > .5

        # Submit the problems
        samplesets: typing.List[dimod.SampleSet] = []
        flipped = np.zeros(bqm.num_variables, dtype=bool)  # what variables are currently flipped
        for i in range(num_spin_reversal_transforms):
            # determine what needs to be flipped
            transform = flipped != SRT[i, :]

            # apply the transform
            for v, flip in zip(bqm.variables, transform):
                if flip:
                    bqm.flip_variable(v)
            flipped[transform] = ~flipped[transform]
            sampleset = sampler.sample(bqm, **kwargs)
            samplesets.append(sampleset)

        # Yield a view of the samplesets that reports done()-ness
        yield self._SampleSets(samplesets)

        # Undo the SRTs according to vartype
        if bqm.vartype is dimod.Vartype.BINARY:
            for i, sampleset in enumerate(samplesets):
                sampleset.record.sample[:, SRT[i, :]] = 1 - sampleset.record.sample[:, SRT[i, :]]
        elif bqm.vartype is dimod.Vartype.SPIN:
            for i, sampleset in enumerate(samplesets):
                sampleset.record.sample[:, SRT[i, :]] *= -1
        else:
            raise RuntimeError("unexpected vartype")

        # Preserves embedding_context and scalar which is the same for all
        # {'embedding_context', 'problem_id', 'problem_label', 'scalar', 'timing'}
        info = {'scalar':samplesets[0].info['scalar'],
                'embedding_context':samplesets[0].info['embedding_context']}

        # finally combine all samplesets together
        response = dimod.concatenate(samplesets)
        response.info.update(info)
        yield response
