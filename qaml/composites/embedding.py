import copy
import dimod

import dwave.embedding

import networkx as nx

class LenientFixedEmbeddingComposite(dimod.ComposedSampler):

    parameters = None
    children = None
    properties = None

    embedding_orig = None
    disjoint_chains : dict

    def __init__(self, child, embedding, scale_aware=False):

        self.children = [child]

        self.parameters = parameters = child.parameters.copy()
        parameters.update(chain_strength=[],
                          chain_break_method=[],
                          chain_break_fraction=[])

        self.properties = dict(child_properties=child.properties.copy())

        self.target_structure = dimod.child_structure_dfs(child)

        self.scale_aware = bool(scale_aware)

        self.embedding_orig = embedding
        self.embedding = self.disjoint(embedding)

    def disjoint(self,embedding):

        self.embedding_orig = copy.deepcopy(embedding)

        # Tried target_structure.to_networkx_graph() but didn't work
        target_graph = nx.Graph(self.target_structure.edgelist)
        target_graph.add_nodes_from(self.target_structure.nodelist)
        self.networkx_graph = target_graph

        # Create "sub-chains" to allow gaps in chains
        new_embedding = {}
        self.disjoint_chains = {}
        for x in self.embedding_orig:
            emb_x = self.embedding_orig[x]
            chain_edges = self.embedding_orig._chain_edges[x]
            # Very inneficient but does the job of creating chain subgraphs
            chain_graph = nx.Graph()
            chain_graph.add_nodes_from([v for v in emb_x if target_graph.has_node(v)])
            chain_graph.add_edges_from([(emb_x[i],emb_x[j]) for i,j in chain_edges if target_graph.has_edge(emb_x[i],emb_x[j])])
            chain_subgraphs = [chain_graph.subgraph(c) for c in nx.connected_components(chain_graph)]

            if len(chain_subgraphs)>1:
                total_chain = sum(len(G) for G in chain_subgraphs)
                for i,G in enumerate(chain_subgraphs):
                    new_embedding[f'{x}_ADACHI_SUBCHAIN_{i}'] = tuple(G.nodes)
                self.disjoint_chains[x] = len(chain_subgraphs)
            elif len(chain_subgraphs)==1:
                new_embedding[x] = list(chain_graph.nodes)
            else:
                raise RuntimeError(f"No subgraphs were found for chain: {x}")
        return dwave.embedding.EmbeddedStructure(target_graph.edges,new_embedding)

    def sample(self, bqm, chain_strength=None,
               chain_break_method=None,
               chain_break_fraction=True,
               return_embedding=None,
               **parameters):

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = self.target_structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        bqm_embedded = self.embed_bqm(bqm,self.embedding,chain_strength,dimod.SPIN)

        if self.scale_aware and 'ignored_interactions' in self.child.parameters:

            ignored = []
            for chain in self.embedding.values():
                # just use 0 as a null value because we don't actually need
                # the biases, just the interactions
                ignored.extend(dwave.embedding.chain_to_quadratic(chain, target_adjacency, 0))

            parameters['ignored_interactions'] = ignored

        response = self.child.sample(bqm_embedded, **parameters)

        def async_unembed(response):

            sampleset = self.unembed_sampleset(response, self.embedding, bqm,
                                          chain_break_method=chain_break_method,
                                          chain_break_fraction=chain_break_fraction,
                                          return_embedding=True)

            if return_embedding:
                sampleset.info['embedding_context'].update(
                    chain_strength  = chain_strength,
                    disjoint_chains = self.disjoint_chains,
                    embedding_orig  = self.embedding_orig)

            return sampleset

        return dimod.SampleSet.from_future(response, async_unembed)

    def embed_bqm(self, bqm, embedding, chain_strength, smear_vartype):

        # Create new BQM including new subchains
        target_bqm = dimod.BinaryQuadraticModel.empty(smear_vartype)
        target_bqm.offset += bqm.offset
        for v, bias in bqm.linear.items():
            if v in embedding:
                chain = embedding[v]
                b = bias / len(chain)
                target_bqm.add_variables_from({u: b for u in chain})
                for p, q in embedding.chain_edges(v):
                    target_bqm.add_interaction(p, q, -chain_strength)
                    target_bqm.offset += chain_strength
            elif v in self.disjoint_chains:
                disjoint_chain = []
                chains = self.disjoint_chains[v]
                for i in range(chains):
                    v_sub = f'{v}_ADACHI_SUBCHAIN_{i}'
                    disjoint_chain += [q for q in embedding[v_sub]]
                    for p, q in embedding.chain_edges(v_sub):
                        target_bqm.add_interaction(p, q, -chain_strength)
                        target_bqm.offset += chain_strength
                b = bias / len(disjoint_chain)
                target_bqm.add_variables_from({u: b for u in disjoint_chain})
            else:
                raise dimod.MissingChainError(v)

        for (u, v), bias in bqm.quadratic.items():
            if u in self.disjoint_chains:
                iter_u = [f'{u}_ADACHI_SUBCHAIN_{i}' for i in range(self.disjoint_chains[u])]
            else:
                iter_u = [u]
            if v in self.disjoint_chains:
                iter_v = [f'{v}_ADACHI_SUBCHAIN_{i}' for i in range(self.disjoint_chains[v])]
            else:
                iter_v = [v]

            interactions = []
            for i_u in iter_u:
                for i_v in iter_v:
                    interactions+=list(embedding.interaction_edges(i_u,i_v))

            b = bias / len(interactions)
            target_bqm.add_interactions_from((u,v,b) for u,v in interactions)

        return target_bqm

    def unembed_sampleset(self, response, embedding, bqm, **unembed_kwargs):
        pruned_embedding = {v:list(q for q in chain if q in self.networkx_graph)
                            for v,chain in self.embedding_orig.items()}

        # Updates sampleset.info with {embedding, chain_break_method}
        sampleset = dwave.embedding.unembed_sampleset(response,pruned_embedding,
                                                     bqm,**unembed_kwargs)

        return sampleset
