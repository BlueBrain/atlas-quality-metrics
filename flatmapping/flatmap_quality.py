import numpy
from scipy.stats import binom, wasserstein_distance, chi2_contingency
from scipy.spatial import distance

import networkx
from networkx.algorithms import connected_components

from .read_atlas_flatmap import FX, FY, LAYER, IX, IY, IZ

def __binom_evaluator__(N):
    def func(p, n):
        if numpy.isnan(n):
            n = 0
        distr = binom(N, p)
        p_left = distr.cdf(n)
        p_right = 1.0 - distr.cdf(n - 1)
        return numpy.minimum(p_left, p_right)
    return func

def wasserstein_wrapper(series_a, series_b):
    return wasserstein_distance(series_a.index.values, series_b.index.values, series_a.values, series_b.values)

def gini_coefficient(values):
    values = sorted(values)[-1::-1]  # large values first
    y = numpy.cumsum(values) / numpy.sum(values)
    return numpy.mean(y) - 0.5

def connected_components_analysis(smpl, max_dist=1):
    M = distance.squareform(distance.pdist(smpl[[IX, IY, IZ]]))
    N = networkx.convert_matrix.from_numpy_array(M <= max_dist)
    L = [len(_x) for _x in connected_components(N)]
    return float(numpy.sum(L) - numpy.max(L)) / numpy.sum(L)

def equal_voxels(df):
    L = df.groupby([FX, FY])[LAYER].agg(len)
    if (-1, -1) in L:
        L = L.drop((-1, -1))
    return gini_coefficient(L.values)

def equal_voxels_per_pixel(df):
    L = df.groupby([FX, FY])[LAYER].agg(len)
    if (-1, -1) in L:
        L = L.drop((-1, -1))
    return (L - L.mean()) / (L + L.mean())

def equal_layers_per_pixel(df):
    tgt_distr = df[LAYER].value_counts() / len(df)
    vc = df.groupby([FX, FY])[LAYER].value_counts()
    if (-1, -1) in vc:
        vc = vc.drop((-1, -1))
    res = vc.groupby([FX, FY]).apply(lambda e: wasserstein_wrapper(e.droplevel([FX, FY]), tgt_distr))
    return res

def equal_layers(df):
    vc = df.groupby([FX, FY])[LAYER].value_counts()
    if (-1, -1) in vc:
        vc = vc.drop((-1, -1))
    vc_table = vc.unstack("layer", fill_value=0)
    _, p_value, _, _ = chi2_contingency(vc_table)
    return -numpy.log10(p_value)

def connected_reverse_image_per_pixel(df, max_dist=1):
    df = df.set_index([FX, FY])
    if (-1, -1) in df.index:
        df = df.drop((-1, -1))
    con_idx = df.groupby([FX, FY]).apply(lambda _x: connected_components_analysis(_x, max_dist=max_dist))
    return con_idx

def connected_reverse_image(df, thresh=0.05, tgt_frac=0.9, epsilon=0.025):
    lower = 1.0
    upper = 10.0
    res = (connected_reverse_image_per_pixel(df, max_dist=upper) <= thresh).mean()
    if res < tgt_frac:
        return upper
    while (upper - lower) > 0.1:  # Half-assed binary search
        to_test = 0.5 * lower + 0.5 * upper
        print("Evaluating: {0}".format(to_test))
        res = (connected_reverse_image_per_pixel(df, max_dist=to_test) <= thresh).mean()
        if numpy.abs(res - tgt_frac) <= epsilon: return to_test
        elif res < tgt_frac: lower = to_test
        else: upper = to_test
    return 0.5 * lower + 0.5 * upper
