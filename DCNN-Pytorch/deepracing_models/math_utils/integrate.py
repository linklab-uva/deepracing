import torch
def cumtrapz(y,x,initial=None):
    dx = x[:,1:]-x[:,:-1]
    avgy = 0.5*(y[:,1:]+y[:,:-1])
    #print("dx shape: ", dx.shape)
    #print("avgy shape: ", avgy.shape)
    mul = avgy*dx[:,:,None]
   # print("mul shape: ", mul.shape)
    res = torch.cumsum(mul, dim=1)
    #res = torch.stack([torch.cumsum(mul[:,:,i],dim=1) for i in range(y.shape[2])],dim=2)
    #print("res shape: ", res.shape)
    if initial is None:
        return res
    return torch.cat((initial[:,None],res),dim=1)
