import numpy as np
import numpy.linalg as la
import scipy, scipy.integrate
from PIL import Image as PILImage
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import json
from functools import reduce
from deepracing.path_utils.optimization import OptimWrapper
import deepracing.path_utils.geometric as geometric
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing




parser = argparse.ArgumentParser()
parser.add_argument("trackfile", help="Path to trackfile to convert",  type=str)
parser.add_argument("ds", type=float, help="Sample the path at points this distance apart along the path")
parser.add_argument("--maxiter", type=float, default=20, help="Maximum iterations to run the solver")
parser.add_argument("--k", default=3, type=int, help="Degree of spline interpolation, ignored if num_samples is 0")
parser.add_argument("--maxv", default=86.0, type=float, help="Max linear speed the car can have")
parser.add_argument("--maxa", default=15.0, type=float, help="Max linear acceleration the car can have")
parser.add_argument("--maxacent", default=22.5, type=float, help="Max centripetal acceleration the car can have")
parser.add_argument("--method", default="SLSQP", type=str, help="Optimization method to use")
parser.add_argument("--out", default=None, type=str, help="Where to put the output file. Default is the same directory as the input .track file")
#parser.add_argument("--negate_normals", action="store_true", help="Flip the sign all all of the computed normal vectors")
args = parser.parse_args()
argdict = vars(args)

# if argdict["negate_normals"]:
#     normalsign = -1.0
# else:
#     normalsign = 1.0
trackfilein = os.path.abspath(argdict["trackfile"])
isracingline = "racingline" in os.path.basename(trackfilein)
isinnerboundary = "innerlimit" in os.path.basename(trackfilein)
outerboundary = not (isracingline or isinnerboundary)


trackdir = os.path.abspath(os.path.dirname(trackfilein))
print("trackdir: %s" %(trackdir,))
ds = argdict["ds"]
k = argdict["k"]
trackin = np.loadtxt(trackfilein,delimiter=",",skiprows=2)

I = np.argsort(trackin[:,0])
track = trackin[I].copy()
r = track[:,0].copy()



Xin = np.zeros((track.shape[0],4))
Xin[:,1] = track[:,1]
Xin[:,2] = track[:,3]
Xin[:,3] = track[:,2]
Xin[:,0] = r
Xin[:,0] = Xin[:,0] - Xin[0,0]
final_vector = Xin[0,1:] - Xin[-1,1:]
final_distance = np.linalg.norm(final_vector)
print("initial final distance: %f" %(final_distance,))
final_unit_vector = final_vector/final_distance
if final_distance>ds:
    extra_distance = final_distance-ds
    nstretch = 4
    rstretch =  np.linspace(extra_distance/final_distance, final_distance - ds, nstretch)
    # rstretch =  np.linspace(final_distance/nstretch,((nstretch-1)/nstretch)*final_distance,nstretch)
    final_stretch = np.row_stack([Xin[-1,1:] + rstretch[i]*final_unit_vector for i in range(rstretch.shape[0])])
    final_r =  rstretch + Xin[-1,0]
    Xin = np.row_stack((Xin, np.column_stack((final_r,final_stretch))))
    final_vector = Xin[0,1:] - Xin[-1,1:]
    final_distance = np.linalg.norm(final_vector)
print("final final distance: %f" %(final_distance,))

fig2 = plt.figure()
plt.plot(Xin[0,1], Xin[0,3], 'g*')
plt.plot(Xin[:,1], Xin[:,3])
plt.title("Input Trackfile")
plt.show()
# rnormalized = Xin[:,0] - Xin[0,0]
# rnormalized = rnormalized/rnormalized[-1]

#bc_type=([(3, np.zeros(3))], [(3, np.zeros(3))])
bc_type=None
# bc_type="natural"

finalidx = -1
if finalidx is None:
    finalextrassamps = 0
else:
    finalextrassamps = abs(finalidx)

rin = Xin[:,0].copy()
rin = rin-rin[0]
rsamp = np.linspace(rin[0], rin[-1], num = int(round((Xin[-1,0]- Xin[0,0])/ds)))


ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
if isinnerboundary:
    ref*=-1.0
spline, Xsamp, unit_tangents, unit_normals = geometric.computeTangentsAndNormals(rin, Xin[:,1:], k=k, rsamp=rsamp, ref=ref)
tangentspline : scipy.interpolate.BSpline = spline.derivative(nu=1)
accelspline : scipy.interpolate.BSpline = spline.derivative(nu=2)


lr = LinearRing([(Xin[i,1], Xin[i,3]) for i in range(0,Xin.shape[0])])
polygon : Polygon = Polygon(lr)
#assert(polygon.is_valid)


tangents = tangentspline(rsamp)
#print(tangents.shape)
tangentnorms = np.linalg.norm(tangents, ord=2, axis=1)
unit_tangents = tangents/tangentnorms[:,np.newaxis]


accels = accelspline(rsamp)
#print(tangents.shape)
accelnorms = np.linalg.norm(accels, ord=2, axis=1)
unit_accels = accelnorms/accelnorms[:,np.newaxis]


# v1 = np.cross(unit_tangents, ref)
# v1 = v1/np.linalg.norm(v1, axis=1, ord=2)[:,np.newaxis]
# v2 =  np.cross(v1, unit_tangents)
# v2 = v2/np.linalg.norm(v2, axis=1, ord=2)[:,np.newaxis]

# unit_normals = np.cross(v2, unit_tangents)
# unit_normals = unit_normals/np.linalg.norm(unit_normals, axis=1, ord=2)[:,np.newaxis]


normaltangentdots = np.sum(unit_tangents*unit_normals, axis=1)
# print(normaltangentdots)
# print(normaltangentdots.shape)
if not np.all(np.abs(normaltangentdots)<=1E-6):
    raise ValueError("Something went wrong. one of the tangents is not normal to it's corresponding normal.")

print("Max dot between normals and tangents: %f" % (np.max(normaltangentdots),) )


offset_points = Xsamp + unit_normals*1.0


xin = Xin[:,1]
yin = Xin[:,2]
zin = Xin[:,3]

xsamp = Xsamp[:,0]
ysamp = Xsamp[:,1]
zsamp = Xsamp[:,2]

diffs = Xsamp[1:] - Xsamp[:-1]
diffnorms = np.linalg.norm(diffs,axis=1)

fig2 = plt.figure()
plt.xlim(np.max(xsamp)+10, np.min(xsamp)-10)
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, c='r', marker='o', s =2.0*np.ones_like(x))
# ax.quiver(x, y, z, unit_normals[:,0], unit_normals[:,1], unit_normals[:,2], length=50.0, normalize=True)
plt.scatter(xin, zin, c='b', marker='o', s = 16.0*np.ones_like(Xin[:,1]))
# plt.scatter(x, z, c='r', marker='o', s = 4.0*np.ones_like(x))
plt.plot(xsamp, zsamp, 'r')
plt.plot(xsamp[0], zsamp[0], 'g*')
plt.quiver(xsamp, zsamp, unit_normals[:,0], unit_normals[:,2], angles="xy", scale=4.0, scale_units="inches")
try:
    plt.show()
except:
    plt.close()

print("Output shape: %s" %(str(Xsamp.shape),))

jsonout = argdict["out"]
if jsonout is None:
    jsonout = os.path.abspath(os.path.join(trackdir,os.path.splitext(os.path.basename(trackfilein))[0] + ".json"))

print("jsonout: %s" %(jsonout,))
if isinnerboundary or outerboundary:
    jsondict = dict()
    jsondict["rin"] = rin.tolist()
    jsondict["xin"] = xin.tolist()
    jsondict["yin"] = yin.tolist()
    jsondict["zin"] = zin.tolist()
    jsondict["r"] = rsamp.tolist()
    jsondict["x"] = xsamp.tolist()
    jsondict["y"] = ysamp.tolist()
    jsondict["z"] = zsamp.tolist()
    jsondict["k"] = k
    jsondict["ds"] = ds
    
    with open(jsonout,"w") as f:
        json.dump( jsondict , f , indent=1 )
    exit(0)


#rsampradii = np.arange(rsamp[0], rsamp[-1]+ds, step=ds)
#rsampradii = rsamp
# rsampradii = np.linspace(rsamp[0], rsamp[-1], num=num_samples)
#ds = rsampradii[1]-rsampradii[0]
print("Optimizing over a space of size: %d" %(rsamp.shape[0],), flush=True)


# tangentsradii = tangentspline(rsampradii)
# tangentsradiinorms = np.linalg.norm(tangentsradii, ord=2, axis=1)
# unittangentsradii = tangentsradii/tangentsradiinorms[:,np.newaxis]


# accels = normalspline(rsampradii)
# accelnorms = np.linalg.norm(accels, ord=2, axis=1)

dotsquares = np.sum(tangents*accels, axis=1)**2

radii = (tangentnorms**3)/(np.sqrt((tangentnorms**2)*(accelnorms**2) - dotsquares) + 1E-6)
radii[0:int(round(92.0/ds))] = np.inf
radii[-int(round(20.0/ds)):] = np.inf
# radii[-2] = 0.5*(radii[-3] + radii[-1])
#radii[-1] = 0.5*(radii[-2] + radii[0])
radii[0] = radii[1]
radii[-1] = radii[-2]

rprint = 50
print("First %d radii:\n%s" %(rprint, str(radii[0:rprint]),))
print("Final %d radii:\n%s" %(rprint, str(radii[-rprint:]),))



maxspeed = argdict["maxv"]
maxlinearaccel = argdict["maxa"]
maxcentripetalaccel = argdict["maxacent"]
dsvec = np.array((rsamp[1:] - rsamp[:-1]).tolist() + [np.linalg.norm(Xsamp[-1] - Xsamp[0])])
#dsvec[-int(round(40/ds)):] = np.inf
print("Final %d delta s:\n%s" %(rprint, str(dsvec[-rprint:]),))
sqp = OptimWrapper(maxspeed, maxlinearaccel, maxcentripetalaccel, dsvec, radii)


#method="trust-constr"
method=argdict["method"]
maxiter=argdict["maxiter"]
x0, res = sqp.optimize(maxiter=maxiter,method=method,disp=True, keep_feasible=True)#,eps=100.0)
print(vars(res), flush=True)
v0 = np.sqrt(x0)
velsquares = res.x
vels = np.sqrt(velsquares)
#vels[-1] = vels[-2]
# resdict = vars(res)
# print(resdict["x"])
# print({key:resdict[key] for key in resdict.keys() if key!="x"})
print(v0, flush=True)
print(vels, flush=True)

velinv = 1.0/vels
#print(velinv)
invspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(rsamp, velinv, k=2)
invsplinead : scipy.interpolate.BSpline = invspline.antiderivative()
tparameterized = invsplinead(rsamp)
tparameterized = tparameterized - tparameterized[0]
positionsradii = spline(rsamp)

truespline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(tparameterized, positionsradii, bc_type=bc_type)
truesplinevel : scipy.interpolate.BSpline = truespline.derivative(nu=1)
truesplineaccel : scipy.interpolate.BSpline = truespline.derivative(nu=2)




nout = 4000
tsampcheck = np.linspace(tparameterized[0], tparameterized[-1], num=radii.shape[0])
tsamp = np.linspace(tparameterized[0], tparameterized[-1], num=nout)
dsamp = np.linspace(rsamp[0], rsamp[-1], num=nout)

splinevels = truesplinevel(tsampcheck)
splinecentripetaccels = np.sum(np.square(splinevels), axis=1)/radii
splinelinearaccels = truesplineaccel(tsampcheck)
print("dt: %f" % (tsamp[-1] - tsamp[0],), flush=True)
print("ds: %f" % (dsamp[-1] - dsamp[0],), flush=True)
print("max centripetal acceleration: %f" % (np.max(splinecentripetaccels)), flush=True)
print("max linear acceleration: %f" % (np.max(splinelinearaccels)), flush=True)

psamp = truespline(tsamp)
xtrue = psamp[:,0]
ytrue = psamp[:,1]
ztrue = psamp[:,2]
final_stretch_samp = psamp[0] - psamp[-1]
print("Final position distance: %f" % (np.linalg.norm(final_stretch_samp, ord=2),), flush=True)

#print("Specified vels minus spline vels:\n" + str(unit_tangents*vels[:,np.newaxis]-truesplinevel(tparameterized)), flush=True)

fig2 = plt.figure()
plt.xlim(np.max(xtrue)+10, np.min(xtrue)-10)
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, c='r', marker='o', s =2.0*np.ones_like(x))
# ax.quiver(x, y, z, unit_normals[:,0], unit_normals[:,1], unit_normals[:,2], length=50.0, normalize=True)
plt.plot(positionsradii[:,0],positionsradii[:,2],'r')
plt.scatter(xtrue[1:], ztrue[1:], c='b', marker='o', s = 16.0*np.ones_like(xtrue[1:]))
plt.plot(xtrue[0], ztrue[0], 'g*')
plt.show()
# try:
# except:
#     plt.close()



# print(vels.shape)
# print(tparameterized.shape)






jsondict : dict = {}
jsondict["r"] = rsamp.tolist()
jsondict["t"] = tparameterized.tolist()
jsondict["x"] = positionsradii[:,0].tolist()
jsondict["y"] = positionsradii[:,1].tolist()
jsondict["z"] = positionsradii[:,2].tolist()
jsondict["rin"] = rin.tolist()
jsondict["xin"] = xin.tolist()
jsondict["yin"] = yin.tolist()
jsondict["zin"] = zin.tolist()
jsondict.update({key : argdict[key] for key in ["maxv", "maxa", "maxacent", "method", "k", "ds"]})
assert(len(jsondict["r"]) == len(jsondict["t"]) == len(jsondict["x"]) == len(jsondict["y"]) == len(jsondict["z"]))


with open(jsonout,"w") as f:
    json.dump( jsondict , f , indent=1 )
    

