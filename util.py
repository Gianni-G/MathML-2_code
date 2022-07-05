# GENERAL FUNCTIONS #
#####################

# Load Modules

import os
import json
import functools
import numpy as np
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

sns.set(
    rc = {'figure.figsize':(25,20)},
    font="Courier"
    )

def plot(vector,
    xticklabels = None,
    yticklabels = None,
    vmin = None,
    vmax = None,
    labelbottom = False,
    labelright = False,
    save = None,

    ):
    hm = sns.heatmap(
        vector,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        linewidths=.5,
        cmap="coolwarm",#"YlGnBu",
        center = 0,
        vmin = vmin,
        vmax = vmax,
        square=True,
        ).tick_params(
            axis='both',
            which='major',
            labelsize=11,
            labelbottom = labelbottom,
            labelright = labelright, 
            bottom=False, 
            top = False, 
            labeltop=True)
    
    plt.yticks(rotation=0) 
    if save != None:
        plt.savefig(save)
    return hm

def pmi_matrix(
    terms,
    contexts,
    ng1,
    ng2,
    ng3,
    total,
    context_side = "right",
    independent = False, # Two possibilities: False: empirical prob of t | True: p(t left)*p(t right)
    inf_value = -1,
    ):
    
    matrix = []
    for i in terms:
        if len(i)==1:
            nt = ng1.get(i,0)
            Pt = nt/total
        elif len(i)==2:
            if independent:
                ntl = ng1.get(i[0],0)
                ntr = ng1.get(i[1],0)
                Pt = (ntl/total)*(ntr/total)
            else:
                nt = ng2.get(i,0)
                Pt = nt/total

        row = []
        for j in contexts:
            nc = ng1.get(j,0)
            Pc = nc/total

            tc = i+j if context_side == "right" else j+i
            if len(i)==1:
                ntc = ng2.get(tc,0)
            elif len(i)==2:
                ntc = ng3.get(tc,0)
            
            Ptc = ntc/total
            if Ptc == 0:
                # the value of pmi when Ptc = 0 should be reconsidered (the best would probably be to use min of the matrix or have a variable measure depending on the prob of the term and context)
                pmi = inf_value
            else:
                pmi = np.log(Ptc/(Pt*Pc))/-np.log(Ptc)

            row.append(pmi)
        
        matrix.append(row)

    matrix = np.array(matrix)
    return matrix

def llf(R, elements, id):
    n = len(elements)
    if id < n:
        return elements[id]
    else:
        label = [id]
        while max(label)>=n:
            new_label = []
            for l in label:
                if l<n:
                    new_label.append(l)
                else:
                    new_label.append(int(R[l-n,0]))
                    new_label.append(int(R[l-n,1]))
            label = new_label
        label = "{ " + " ".join([elements[l] for l in label]) + " }"

        return label

def multithreading(func, args, chunksize=1,cores=None):
    with concurrent.futures.ThreadPoolExecutor(cores) as executor:
        result = executor.map(func, args, chunksize=chunksize)
    return list(result)

def load_file(filename):
    
    if not os.path.isfile(filename):
        raise Exception(f"SLG [E]: {filename} does not exist")
    
    fn_extension = os.path.splitext(filename)[-1]

    if fn_extension == ".json":
        with open(filename) as json_file:
            data = json.load(json_file)

    else:
        raise Exception(f"SLG [E]: File extension {fn_extension} not recognised")

    return data

def normalize_dict(dictionnary: dict, norm_factor = None):
    if norm_factor == None:
        norm = 1/sum(dictionnary.values())
    else:
        norm = norm_factor
    prob_dict = {k: v * norm for k, v in dictionnary.items()}
    return prob_dict

def pmi(
    matrix,
    alpha=.75,
    type_pmi="sppmi",
    ):
    """
    """
    # Taken from Kaggle, and modified

    if type_pmi == "nopmi":
        return matrix

    if type_pmi not in {"pmi","npmi","ppmi","nppmi","spmi","sppmi","ssppmi","nopmi"}:
        pmi = "sppmi"
        print("Unknown type of PMI. Continuing with smoothed positive PMI (SPPMI)")

    num_skipgrams = matrix.sum()
    matrix_items = {(i, j): matrix[i,j] for i, j in zip(*matrix.nonzero())}
    # assert(sum(matrix_items.values()) == num_skipgrams)


    # for creating sparse matrices
    row_indxs = []
    col_indxs = []

    pmi_dat_values = []    # pointwise mutual information

    # reusable quantities
    sum_over_contexts = np.array(matrix.sum(axis=1)).flatten()
    sum_over_terms = np.array(matrix.sum(axis=0)).flatten()

    # smoothing
    sum_over_terms_alpha = sum_over_terms**alpha
    nca_denom = np.sum(sum_over_terms_alpha)

    for (tok_terms, tok_context), sg_count in matrix_items.items():
        # here we have the following correspondance with Levy, Goldberg, Dagan
        #========================================================================
        #   num_skipgrams = |D|
        #   nwc = sg_count = #(w,c)
        #   Pwc = nwc / num_skipgrams = #(w,c) / |D|
        #   nw = sum_over_cols[tok_ingredient]    = sum_over_contexts[tok_ingredient] = #(w)
        #   Pw = nw / num_skipgrams = #(w) / |D|
        #   nc = sum_over_rows[tok_context] = sum_over_ingredients[tok_context] = #(c)
        #   Pc = nc / num_skipgrams = #(c) / |D|
        #
        #   nca = sum_over_rows[tok_context]^alpha = sum_over_ingredients[tok_context]^alpha = #(c)^alpha
        #   nca_denom = sum_{tok_content}( sum_over_ingredients[tok_content]^alpha )

        nwc = sg_count
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok_terms]
        Pw = nw / num_skipgrams
        
        # note 
        # pmi = log {#(w,c) |D| / [#(w) #(c)]} 
        #     = log {nwc * num_skipgrams / [nw nc]}
        #     = log {P(w,c) / [P(w) P(c)]} 
        #     = log {Pwc / [Pw Pc]}

        if type_pmi == "pmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = np.log(Pwc/(Pw*Pc))

        elif type_pmi == "npmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = np.log(Pwc/(Pw*Pc))/-np.log(Pwc)
        
        elif type_pmi == "ppmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = max(np.log(Pwc/(Pw*Pc)), 0)

        elif type_pmi == "nppmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = max(np.log(Pwc/(Pw*Pc))/-np.log(Pwc),0)

        elif type_pmi == "spmi":
            nca = sum_over_terms_alpha[tok_context]
            Pca = nca / nca_denom
            pmi = np.log(Pwc/(Pw*Pca))
        elif type_pmi == "sppmi":
            nca = sum_over_terms_alpha[tok_context]
            Pca = nca / nca_denom  
            pmi = max(np.log(Pwc/(Pw*Pca)), 0)
        elif type_pmi == "ssppmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = max(np.log(Pwc/(Pw*Pc*alpha)), 0)
        
        row_indxs.append(tok_terms)
        col_indxs.append(tok_context)
        pmi_dat_values.append(pmi)
            
    pmi_mat = csr_matrix((pmi_dat_values, (row_indxs, col_indxs)),shape=matrix.shape)

    print('Done')
    return pmi_mat



def mm_no_modif(contexts, orthogonals, term):

    if isinstance(next(iter(orthogonals)),str):
        return [orthogonals.get(term+context,0) for context in contexts]
    elif isinstance(next(iter(orthogonals)),tuple):
        return [orthogonals.get((term, context),0) for context in contexts]
    else:
        raise Exception(f"SLG [E]: Orthogonal dictionnary keys have the wrong type ({type(type(next(iter(orthogonals))))}). Types accepted: str and tuple ")

def matrix_maker(
    terms,
    contexts,
    orthogonals,
    measure = mm_no_modif):
    results = multithreading(functools.partial(measure, contexts, orthogonals), terms)
    return results

def build_term_context_matrix(
    terms,
    contexts,
    orthogonals,
    normalizeQ = False):
    print("Building oR Matrix...")
    start = time.perf_counter()
    if normalizeQ:
        orthogonals = normalize_dict(orthogonals)
    matrix = csr_matrix(matrix_maker(terms,contexts,orthogonals))
    finish = time.perf_counter()
    print(f"Term-Context Matrix built in {round(finish-start,2)} secs.\n")
    return matrix

def build_pmi_matrix(
    term_context_matrix,
    type = "pmi",
    alpha = .75,
    normalizeQ = False,
    ):

    print("Computing PMI Matrix...")
    print(f"Type: {type}")
    if "s" in type:
        print(f"Smoothing (alpha): {alpha}")
    start = time.perf_counter()
    pmi_matrix = pmi(term_context_matrix,alpha=alpha,type_pmi=type)
    finish = time.perf_counter()
    if normalizeQ:
        print("Normalizing Matrix")
        pmi_matrix = (1/(pmi_matrix.sum()))*pmi_matrix
    print(f"PMI Matrix built in {round(finish-start,2)} secs.")
    print("Done\n")
    return pmi_matrix