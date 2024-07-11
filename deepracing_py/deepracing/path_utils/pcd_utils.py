

from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation
#From the PCD spec, the valid type characters are: 
#I - represents signed types int8 (char), int16 (short), and int32 (int)
#U - represents unsigned types uint8 (unsigned char), uint16 (unsigned short), uint32 (unsigned int)
#F - represents float types

_NUMPY_TYPEMAP : dict = dict()

_NUMPY_TYPEMAP[("I", 1)] = np.int8
_NUMPY_TYPEMAP[("I", 2)] = np.int16
_NUMPY_TYPEMAP[("I", 4)] = np.int32
_NUMPY_TYPEMAP[("I", 8)] = np.int64
# _NUMPY_TYPEMAP[("I", 16)] = np.int128

_NUMPY_TYPEMAP[("U", 1)] = np.uint8
_NUMPY_TYPEMAP[("U", 2)] = np.uint16
_NUMPY_TYPEMAP[("U", 4)] = np.uint32
_NUMPY_TYPEMAP[("U", 8)] = np.uint64
# _NUMPY_TYPEMAP[("U", 16)] = np.uint128

_NUMPY_TYPEMAP[("F", 4)] = np.float32
_NUMPY_TYPEMAP[("F", 8)] = np.float64


_NUMPY_INVERSE_TYPEMAP : dict = {_NUMPY_TYPEMAP[k] : k for k in _NUMPY_TYPEMAP.keys()}


# _NUMPY_TYPEMAP[("F", 16)] = np.float128

# Valid PCD files have the following header keys in this specific order:
# VERSION
# FIELDS
# SIZE
# TYPE
# COUNT
# WIDTH
# HEIGHT
# VIEWPOINT
# POINTS
# DATA
_VERSION_TAG_LINE = 0
_FIELDS_TAG_LINE = 1
_SIZE_TAG_LINE = 2
_TYPE_TAG_LINE = 3
_COUNT_TAG_LINE = 4
_WIDTH_TAG_LINE = 5
_HEIGHT_TAG_LINE = 6
_VIEWPOINT_TAG_LINE = 7
_POINTS_TAG_LINE =8
_DATA_TAG_LINE = 9
_NUM_PCD_HEADER_LINES=10

#An examples of such a header for an ascii PCD

# VERSION 0.7
# FIELDS x y z distance
# SIZE 4 4 4 4
# TYPE F F F F
# COUNT 1 1 1 1
# WIDTH 2640
# HEIGHT 1
# VIEWPOINT -0.271 0.654 0.653 0.271 -109.206 2.884 463.537
# POINTS 2640
# DATA ascii

def decodePCDHeader(headerlines : list, align=False):
    fields_string = headerlines[_FIELDS_TAG_LINE].replace("FIELDS","").strip()
    fieldnames : list = fields_string.split(" ")
    numfields = len(fieldnames)

    sizes_string : str = headerlines[_SIZE_TAG_LINE].replace("SIZE","").strip()
    sizes : list = [int(s) for s in sizes_string.split(" ")]
    numsizes = len(sizes)

    if numsizes!=numfields:
        raise ValueError("Got FIELDS tag: %s with %d elements, but SIZE tag %s with %d elements" % (fields_string, numfields, sizes_string, numsizes))

    types_string : str = headerlines[_TYPE_TAG_LINE].replace("TYPE","").strip()
    types : list = types_string.split(" ")
    numtypes = len(types)

    if numtypes!=numfields:
        raise ValueError("Got FIELDS tag: %s with %d elements, but TYPE tag %s with %d elements" % (fields_string, numfields, types_string, numtypes))

    counts_string : str = headerlines[_COUNT_TAG_LINE].replace("COUNT","").strip()
    counts : list = [int(s) for s in counts_string.split(" ")]
    numcounts = len(counts)

    if numcounts!=numfields:
        raise ValueError("Got FIELDS tag: %s with %d elements, but COUNT tag %s with %d elements" % (fields_string, numfields, counts_string, numcounts))

    height_string : str = headerlines[_HEIGHT_TAG_LINE].replace("HEIGHT","").strip()
    height : int = int(height_string)

    width_string : str = headerlines[_WIDTH_TAG_LINE].replace("WIDTH","").strip()
    width : int = int(width_string)

    numpoints_string : str = headerlines[_POINTS_TAG_LINE].replace("POINTS","").strip()
    numpoints : int = int(numpoints_string)
    
    if not (numpoints==(height*width)):
        raise ValueError("Got non-dense PCD file with height %d and width %d, but containing %d points. Number of points should equal height*width" % (height, width, numpoints))
        
    numpytuples : list = []
    for i in range(numfields):  
        name = fieldnames[i]
        typestr = types[i]
        size = sizes[i]
        count = counts[i]
        key = (typestr, size)
        if key not in _NUMPY_TYPEMAP.keys():
            raise ValueError("Got invalid combination of type %s and size %d for field %s, only supported combinations are %s" % (typestr, size, name, str(list(_NUMPY_TYPEMAP.keys()))))
        numpytuples.append((name, _NUMPY_TYPEMAP[key], (count,)))
        
    numpytype = np.dtype(numpytuples, align=align)
    return numpytype, height, width

def loadPCD(filepath : str, align=False) -> np.ndarray:

    with open(filepath, "rb") as f:
        headerlines : list = [f.readline().decode("ascii").strip() for asdf in range(_NUM_PCD_HEADER_LINES)]
        data_tag = headerlines[_DATA_TAG_LINE].replace("DATA","").strip() 
        if data_tag not in {"binary", "ascii"}:
            raise ValueError("Invalid DATA tag %s. Supported types are \"ascii\" or \"binary\"" % (data_tag,))
        numpytype, height, width = decodePCDHeader(headerlines, align=(align and (data_tag=="ascii")))
        if data_tag=="binary":
            structured_numpy_array = np.frombuffer(f, dtype=numpytype)
        else:
            structured_numpy_array = np.loadtxt(f, dtype=numpytype, encoding="ascii", delimiter=" ")
    return numpytype, structured_numpy_array, height, width

def structurednumpyToPCD(points : np.ndarray, filepath : str, fmt="%.4f", viewpoint_pos : np.ndarray = np.zeros(3), viewpoint_rot : Rotation = Rotation.identity(), binary=False):
    
    headerlines : list = ["asdf" for asdf in range(_NUM_PCD_HEADER_LINES)]
    headerlines[_VERSION_TAG_LINE] = "VERSION 0.7\n"
    headerlines[_FIELDS_TAG_LINE] = "FIELDS "
    headerlines[_SIZE_TAG_LINE] = "SIZE "
    headerlines[_TYPE_TAG_LINE] = "TYPE "
    headerlines[_COUNT_TAG_LINE] = "COUNT "

    numpytype : np.dtype = points.dtype
    for i in range(len(numpytype.names)):
        name : str = numpytype.names[i]
        headerlines[_FIELDS_TAG_LINE]+=name
        fieldtype, _ = numpytype.fields[name]
        # print(fieldtype)
        subtype, subshape = fieldtype.subdtype
        # print(subtype)
        subtypestring = subtype.str.replace("<","")
        # print(subtypestring)
        headerlines[_TYPE_TAG_LINE]+=subtypestring[0].upper()
        # headerlines[_TYPE_TAG_LINE]+=fieldtype.char.upper()
        
        headerlines[_SIZE_TAG_LINE]+=str(fieldtype.itemsize)
        if fieldtype.fields is None:
            headerlines[_COUNT_TAG_LINE]+="1"
        else:
            headerlines[_COUNT_TAG_LINE]+=str(len(fieldtype.fields))
        if i==len(numpytype.names)-1:
            headerlines[_FIELDS_TAG_LINE]+="\n"
            headerlines[_SIZE_TAG_LINE]+="\n"
            headerlines[_TYPE_TAG_LINE]+="\n"
            headerlines[_COUNT_TAG_LINE]+="\n"
        else:
            headerlines[_FIELDS_TAG_LINE]+=" "
            headerlines[_SIZE_TAG_LINE]+=" "
            headerlines[_TYPE_TAG_LINE]+=" "
            headerlines[_COUNT_TAG_LINE]+=" "
    if points.ndim==1:
        height = 1
        width = points.shape[0]
    else:
        height, width = points.shape[0], points.shape[1]
    headerlines[_WIDTH_TAG_LINE] = "WIDTH %d\n" % (width,)
    headerlines[_HEIGHT_TAG_LINE] = "HEIGHT %d\n" % (height,)
    viewpoint_pose = np.concatenate([viewpoint_rot.as_quat(), viewpoint_pos])
    headerlines[_VIEWPOINT_TAG_LINE] = "VIEWPOINT %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n" % tuple(viewpoint_pose.tolist())
    headerlines[_POINTS_TAG_LINE] = "POINTS %d\n" % (width*height,)
    if binary:
        headerlines[_DATA_TAG_LINE] = "DATA binary\n"
        with open(filepath, "wb") as f:
            f.writelines([l.encode("ascii") for l in headerlines])
            f.write(points.tobytes())
    else:
        headerlines[_DATA_TAG_LINE] = "DATA ascii\n"
        with open(filepath, "w") as f:
            f.writelines(headerlines)
            np.savetxt(f, points, fmt=fmt, delimiter=" ", newline="\n", encoding="ascii")


def numpyToPCD(x : np.ndarray, points : np.ndarray, filepath : str, 
               x_name : str = "time", viewpoint_pos : np.ndarray = np.zeros(3), viewpoint_rot : Rotation = Rotation.identity()):

    viewpoint_pose = np.concatenate([viewpoint_rot.as_quat(), viewpoint_pos])
    with open(filepath, "w") as f:
        headerlines = ["VERSION 0.7\n", 
                       "FIELDS x y z %s\n" % (x_name,), 
                       "SIZE 4 4 4 4\n", 
                       "TYPE F F F F\n", 
                       "COUNT 1 1 1 1\n", 
                       "WIDTH %d\n" % (points.shape[0],),
                       "HEIGHT 1\n",
                       "VIEWPOINT %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n" % tuple(viewpoint_pose.tolist()),
                       "POINTS %d\n" % (points.shape[0],),
                       "DATA ascii\n"]
        f.writelines(headerlines)
        block = np.concatenate([points, x[:,None]], axis=1)
        np.savetxt(f, block, fmt="%.5f")