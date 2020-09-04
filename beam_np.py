import numpy as np
from scipy.special import softmax

EOS_INDEX = 0
VOCAB_SIZE = 100
MAX_SEQ_LEN = 20
k = 30

np.random.seed(563)
transition_table = np.random.randn(MAX_SEQ_LEN, VOCAB_SIZE)
transition_table = softmax(transition_table, axis=1)

pathes = [
    (1.0, [-1, 5, 6, 7])
]

def expand_node(step, p):
    if p[1][-1] == EOS_INDEX:
        return []
    return np.argsort(-transition_table[step])

for step in range(MAX_SEQ_LEN):
    new_pathes = []
    for p in pathes:
        # generowanie następników
        candidates = expand_node(step, p)
        if len(candidates) == 0:
            new_pathes.append(p)
        else:
            for c in candidates:
                patch_score, patch_list = p
                new_score = patch_score * transition_table[step, c]
                new_patch = patch_list + [c]
                new_pathes.append((new_score, new_patch))
    

    new_pathes = sorted(new_pathes, reverse=True)
    new_pathes = new_pathes[:k]
    pathes = new_pathes

pass



    


