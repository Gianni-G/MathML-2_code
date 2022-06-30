# GENERAL FUNCTIONS #
#####################

# Load Modules

import os, sys
import math
import concurrent.futures
from joblib import Parallel, delayed
import csv
import json
import itertools
import functools
import ast
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
import time
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import socket
socket_name = socket.gethostname()
if any(name in socket_name for name in {"Gianni","vpn"}):
    from tqdm.notebook import tqdm, trange
else:
    from tqdm.auto import tqdm, trange

# Definitions of Functions

def chunk_size(total_n, chunks_n):
    return math.ceil(total_n / chunks_n)

def partition(lst, n_of_parts):
    """Partitions list in n parts."""
    if not isinstance(lst,list):
        lst = list(lst)
    chunk_size = math.ceil(len(lst) / n_of_parts)
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

# head and chunks taken from https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it

def head(iterable, max=10):
    first = next(iterable)      # raise exception when depleted
    def head_inner():
        yield first             # yield the extracted first element
        for cnt, el in enumerate(iterable):
            yield el
            if cnt + 1 >= max:  # cnt + 1 to include first
                break
    return head_inner()

def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))
        
def multiprocessing(func, args, cores=None):
    if cores == None:
        cores = os.cpu_count()

    result = Parallel(n_jobs=cores)(delayed(func)(i) for i in args)
    return result
    
    # with concurrent.futures.ProcessPoolExecutor(cores) as executor:
    #     result = executor.map(func, args, chunksize=chunksize)
    # return list(result)

def multithreading(func, args, chunksize=1,cores=None):
    with concurrent.futures.ThreadPoolExecutor(cores) as executor:
        result = executor.map(func, args, chunksize=chunksize)
    return list(result)


# Adapted Dans Shiebler: from http://danshiebler.com/2016-09-14-parallel-progress-bar/

def multiprocessing_tqdm(function, array, cores=None, desc = None):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    # if front_num > 0:
    #     front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    # #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    # if n_jobs==1:
    #     return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as pool:
        #Pass the elements of array into function

        futures = [pool.submit(function, a) for a in array]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(concurrent.futures.as_completed(futures), **kwargs, desc = desc):
            pass
    out = []
    #Get the results from the futures. 
    for future in futures:
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out

def save_file(data, filename):
    
    if os.path.exists(filename):
        raise Exception(f"SLG [E]: {filename} already exist [TODO: implement overwrite]")

    fn_dir = os.path.dirname(filename)
    fn_extension = os.path.splitext(filename)[-1]

    if not os.path.isdir(fn_dir):
        os.makedirs(fn_dir)

    if fn_extension == ".json":
        with open(filename, 'w',encoding='utf-8') as json_file:
            json.dump(data, json_file,indent=4, ensure_ascii=False)

    else:
        raise Exception(f"SLG [E]: File extension {fn_extension} not recognised")



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

def dict2csv(input: dict, filename: str, path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}{filename}.csv", "w") as nf:
        for key in input.keys():
            nf.write(f"{key},{input[key]}\n")

def dict2json(input: dict, filename: str, path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}/{filename}.json", 'w',encoding='utf-8') as json_file:
        json.dump(input, json_file,indent=4, ensure_ascii=False)




def json2dict(filename, path):
    with open(f"{path}/{filename}.json") as json_file:
        data = json.load(json_file)
    return data

def str2txt(input: str, filename: str, path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}{filename}.txt", "w") as nf:
        nf.write(input)

def list2txt(input: list, filename: str = "text_list", path = None):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}/{filename}.txt", "w") as nf:
        for element in input:
            nf.write(element+"\n")
            
def txt2list(filename, path):
    with open(f"{path}/{filename}.txt") as txt_file:
        txt_list = [line.strip() for line in txt_file]
    return txt_list

def subsequences(sequence, n: int):
    list_of_tuples = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    return list_of_tuples

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def list2csv(input: list, filename: str, directory: str):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(f"/{directory}{filename}.csv", "w", newline="") as nf:
        wr = csv.writer(nf, quoting=csv.QUOTE_ALL)
        wr.writerows(input)

def csv2list(file: str, directory: str, n_start=None, n_end=None):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(f"{directory}{file}.csv", "r") as f:
        csv_reader = csv.reader(f)
        my_list = [
            (line[0], ast.literal_eval(line[1]), ast.literal_eval(line[2]))
            for line in take(n_end, csv_reader)
        ]
    return my_list[n_start:n_end]

# TODO: Really needed here?
def if_none_disable(x):
    if x == None:
        x = "disable"
    return x


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(itertools.islice(iterable, n))

def marginalize(joint_dist:dict,side="left"):
    # print(f"Marginalizing to the {side}")
    marginal = Counter()
    for (l,r),v in joint_dist.items():
        marginal[r if side == "right" else l] += v
    return dict(marginal.most_common())

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



def parallel_processes(processes:list):
    """
    Warning: this produces global variables "parallel_process_n" which are deleted at the end of the computation.
    """
    st = time.perf_counter()
    np = len(processes)
    print(f"Computing {np} processes in parallel...")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=np)
    for n, process in zip(range(np),processes):
        variable_name = f"parallel_process_{n}"
        globals()[variable_name] = executor.submit(*process)
    for n in range(np):
        variable_name = f"parallel_process_{n}"
        globals()[variable_name] = eval(variable_name).result()
    fin = time.perf_counter()
    print(f"{np} processes computed in {round(fin-st,2)} secs.\n")
    result = [eval(f"parallel_process_{n}") for n in range(np)]
    for n in range(np):
        del globals()[f"parallel_process_{n}"]
    return result

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def delete_duplicates(x):
    return list(dict.fromkeys(x))

def clear_df(df):
    mask = df.applymap(lambda x: x is None)
    cols = df.columns[(mask).any()]
    for col in df[cols]:
        df.loc[mask[col], col] = ''
    return df

def df(list_of_list, keys = None):
    raw_df = pd.DataFrame(list_of_list,index=keys)
    return clear_df(raw_df.T)

# Draft functions

# def csv2list_raw(file: str, directory: str, n_start=None, n_end=None):
#     if not os.path.isdir(directory):
#         os.makedirs(directory)
#     with open(f"{directory}{file}.csv", "r") as f:
#         csv_reader = csv.reader(f)
#         my_list = [
#             [ast.literal_eval(item) for item in line]
#             for line in take(n_end, csv_reader)
#         ]
#     return my_list[n_start:n_end]

def csv2list_raw(file: str, directory: str, n_start=None, n_end=None,progress=False):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(f"{directory}{file}.csv", "r") as f:
        csv_reader = csv.reader(f)
        my_list = []
        if progress:
            csv_reader_it = tqdm(take(n_end, csv_reader),desc = f"Loading: {file}")
        else:
            csv_reader_it = take(n_end, csv_reader)
        for line in csv_reader_it:
            list_line = [ast.literal_eval(item) for item in line]
            my_list.append(list_line)
    return my_list[n_start:n_end]

def plot_scatter_line(
    x,
    y,
    trace_name = None,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    line_shape='spline',
    add_trace = None
    ):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name = trace_name, line_shape=line_shape, mode='lines+markers'))
    
    if add_trace != None:
        if isinstance(add_trace,tuple):
            add_trace = [add_trace]
        for trace in add_trace:
            fig.add_trace(go.Scatter(x=trace[0], y=trace[1], name = trace[2], line_shape=line_shape, mode='lines+markers')) # TODO: rewrite with **kwargs


    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=xaxis_title,
        
        )
    return fig

def scatter_plot(data:list,labels=None,legends=None,size=(15,10)):
    fig, ax = plt.subplots(figsize=size)  # Create a figure and an axes.
    for dataset in data:
        ax.scatter(range(len(dataset)) if labels==None else labels,dataset,label="Data")
    ax.set_xlabel('x')  # Add an x-label to the axes.
    ax.set_ylabel('f(x)')  # Add a y-label to the axes.
    ax.set_title("Title")  # Add a title to the axes.
    ax.legend(range(1,len(data)+1) if legends==None else legends)  # Add a legend.
    return ax

profiler = """
from pyinstrument import Profiler
import sys
def profiler(process):
    profile = Profiler()

    profile.start()
    
    try:
        exec(process)
    except:
        print(f"{sys.exc_info()}")

    profile.stop()

    return print(profile.output_text(unicode=True, color=True))"""


# value and index of max of matrix, in case needed
# np.array(np.unravel_index(np.argsort(voc_matrix.toarray().flatten(), axis=0)[-10:], voc_matrix.shape)).T[::-1]
