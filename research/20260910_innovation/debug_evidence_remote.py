"""Read-only diagnostic of the remote numerical test failure."""
import json
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0, '/data/sunzixuan/target_isolated_20260910_v1')
from evidence import build_counterfactual_cache

train = np.array([[1,1,0,0],[1,0,1,0],[0,1,1,1]], dtype=float)
record = {'numpy': np.__version__}
for label, matrix in [('dense',train),('csr',sparse.csr_matrix(train))]:
    cache = build_counterfactual_cache(matrix, topk=2)
    row = {'similarity': (cache.similarity.toarray() if label=='csr' else cache.similarity).tolist(),
           'raw_scores': cache.raw_scores.tolist(), 'raw_dtype': str(cache.raw_scores.dtype)}
    for field in ('edge_users','edge_items','original_mean','original_std', 'history_mean','history_std',
                  'edge_mean','edge_std','edge_target_raw','self_vote_raw'):
        row[field] = getattr(cache,field).tolist()
    row['pairs'] = {mode:[x.tolist() for x in cache.get_pair([0,1,2],[0,2,3],[2,3,0],mode)]
                    for mode in ('original','history','edge')}
    record[label] = row
print(json.dumps(record,indent=2))
