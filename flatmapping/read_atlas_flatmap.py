import nrrd
import os
import numpy
import pandas
import json

X = "x"
Y = "y"
Z = "z"

IX = "ix"
IY = "iy"
IZ = "iz"

FX = "fx"
FY = "fy"

ANN = "region_id"
LAYER = "layer"

def __use_root__(fn, at_root):
    if at_root is not None:
        if not os.path.isabs(fn):
            fn = os.path.join(at_root, fn)
    return fn

def __parse_layer_str_or_try_to__(a_str):
    if a_str is None:
        return -1
    if ";L" in a_str:
        return int(a_str.split(";L")[1])
    return 0

def __id_to_acronym__(hier, tgt_id):
    if hier["id"] == tgt_id:
        return hier["acronym"]
    for child in hier.get("children", []):
        cand = __id_to_acronym__(child, tgt_id)
        if cand is not None: return cand
    return None

def __get_relevant_direction_rows__(meta):
    valid = numpy.ones(meta["dimension"], dtype=bool)
    if "space dimension" in meta:
        valid[numpy.arange(meta["dimension"] - meta["space dimension"])] = False
    return meta["space directions"][valid]

def __test_volume_compatibility__(meta1, meta2):
    if "space dimension" in meta1:
        assert meta1["space dimension"] == 3, "Only for 3d volumes!"
    else:
        assert meta1["dimension"] == 3
    if "space dimension" in meta2:
        assert meta2["space dimension"] == 3, "Only for 3d volumes!"
    else:
        assert meta2["dimension"] == 3
    dir1 = __get_relevant_direction_rows__(meta1)
    dir2 = __get_relevant_direction_rows__(meta2)
    # Taken out because of slight numerical differences... TODO: Find better way
    # assert numpy.all(meta1["space origin"] == meta2["space origin"]), "Volumes incompatible!"
    # assert numpy.all(dir1 == dir2), "Volumes incompatible!"

def atlas_flatmap_as_dataframe(fn_ann, fn_fm, fn_hier, at_root=None):
    fn_ann = __use_root__(fn_ann, at_root)
    fn_fm = __use_root__(fn_fm, at_root)
    fn_hier = __use_root__(fn_hier, at_root)
    vol_ann, meta_ann = nrrd.read(fn_ann)
    vol_fm, meta_fm = nrrd.read(fn_fm)
    __test_volume_compatibility__(meta_ann, meta_fm)

    with open(fn_hier, "r") as fid:
        hier = json.load(fid)

    mask = vol_ann > 0
    ix, iy, iz = numpy.nonzero(mask)
    x, y, z = numpy.dot(meta_ann["space directions"], numpy.vstack([ix, iy, iz]))
    fx, fy = vol_fm[:, mask]
    ann = vol_ann[mask]

    df = pandas.DataFrame.from_dict(
        {
            X: x,
            Y: y,
            Z: z,
            IX: ix,
            IY: iy,
            IZ: iz,
            ANN: ann,
            FX: fx,
            FY: fy
        }
    )
    u_regions = df[ANN].drop_duplicates()
    layer_map = u_regions.apply(lambda x: __parse_layer_str_or_try_to__(__id_to_acronym__(hier, x)))
    layer_map.index = u_regions
    df[LAYER] = layer_map[df[ANN]].values
    return df
