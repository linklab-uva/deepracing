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
    return torch.cat((initial,res),dim=1)
#come back to this later
# def simpson(f_x, x, simpsonintervals=4, d0=0.0):
#     simpsonscale = torch.ones(f_x.shape[0], simpsonintervals+1, 1, dtype=f_x.dtype, device=f_x.device)
#     simpsonscale[:,[i for i in range(1,simpsonintervals,2)]] = 4.0
#     simpsonscale[:,[i for i in range(2,simpsonintervals,2)]] = 2.0
#     Vmat = torch.stack([ torch.stack([speeds[i,4*j:4*j+5] for j in range(0, N)], dim=0)   for i in range(f_x.shape[0])], dim=0)
#     relmoves = torch.matmul(Vmat, simpsonscale)[:,:,0]
#     distances = torch.cat([d0*torch.ones(f_x.shape[0],1,dtype=f_x.dtype, device=f_x.device),torch.cumsum(relmoves,dim=1)/(3.0*simpsonintervals*N)],dim=1)