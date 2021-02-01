import torch
def cumtrapz(y,x,initial=None):
    dx = x[:,1:]-x[:,:-1]
    avgy = 0.5*(y[:,1:]+y[:,:-1])
    #print("dx shape: ", dx.shape)
    #print("avgy shape: ", avgy.shape)
    mul = avgy*dx
   # print("mul shape: ", mul.shape)
    res = torch.cumsum(mul,1)
    #res = torch.stack([torch.cumsum(mul[:,:,i],dim=1) for i in range(y.shape[2])],dim=2)
    #print("res shape: ", res.shape)
    if initial is None:
        return res
    return torch.cat([initial,res],dim=1)
#come back to this later
def simpson(f_x, delta_x):
    numpoints = f_x.shape[1]
    if numpoints%2==0:
        raise ValueError("Number of points in f_x must be odd (for an even number of intervals as required by simpsons method)")
    if delta_x.shape[0]!=f_x.shape[0]:
        raise ValueError("Batch size of %d for delta_x but batch size of %d for f_x" %(delta_x.shape[0], f_x.shape[0]))
    simpsonintervals = numpoints -1

    simpsonscale = torch.ones(f_x.shape[0], numpoints, dtype=f_x.dtype, device=f_x.device)
    simpsonscale[:,list(range(1,simpsonintervals,2))] = 4.0
    simpsonscale[:,list(range(2,simpsonintervals,2))] = 2.0
    
    return (delta_x/3.0)*torch.sum(simpsonscale*f_x, dim=1)