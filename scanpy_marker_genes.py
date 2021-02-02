#!/usr/bin/env python

import scanpy as sc
import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from granatum_sdk import Granatum


def main():

    tic = time.perf_counter()

    gn = Granatum()

    assay = gn.get_import('assay')
    sample_ids = assay.get('sampleIds')
    group_dict = gn.get_import('groupVec')
    group_vec = pd.Categorical([group_dict.get(x) for x in sample_ids])
    num_groups = len(group_vec.categories)
    figheight = 400 * (math.floor((num_groups - 1) / 7) + 1)

    adata = sc.AnnData(np.array(assay.get('matrix')).transpose())
    adata.var_names = assay.get('geneIds')
    adata.obs_names = assay.get('sampleIds')
    adata.obs['groupVec'] = group_vec

    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X', method='gauss')
    
    try:

        sc.tl.rank_genes_groups(adata, 'groupVec', n_genes=100000)
        sc.pl.rank_genes_groups(adata, n_genes=20)
        gn.add_current_figure_to_results('One-vs-rest marker genes', dpi=75, height=figheight)

        gn._pickle(adata, 'adata')

        rg_res = adata.uns['rank_genes_groups']

        for group in rg_res['names'].dtype.names:
            genes_names = [str(x[group]) for x in rg_res['names']]
            scores = [float(x[group]) for x in rg_res['scores']]
            newdict = dict(zip(genes_names, scores))
            gn.export(newdict, 'Marker score ({} vs. rest)'.format(group), kind='geneMeta')
            gn.export(pd.DataFrame(newdict).to_csv(), 'Marker score {} vs rest.csv'.format(group), kind='raw', meta=None, raw=True)

        # cluster_assignment = dict(zip(adata.obs_names, adata.obs['louvain'].values.tolist()))
        # gn.export_statically(cluster_assignment, 'cluster_assignment')

        toc = time.perf_counter()
        time_passed = round(toc - tic, 2)

        timing = "* Finished marker gene identification step in {} seconds*".format(time_passed)
        gn.add_result(timing, "markdown")
        
        gn.commit()

    except Exception as e:

        plt.figure()
        plt.text(0.01, 0.5, 'Incompatible group vector due to insufficent cells')
        plt.text(0.01, 0.3, 'Please retry the step with a different group vector')
        plt.axis('off')
        gn.add_current_figure_to_results('One-vs-rest marker genes')

        gn.commit()



if __name__ == '__main__':
    main()
