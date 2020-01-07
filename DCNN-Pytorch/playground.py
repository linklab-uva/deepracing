import torch
import deepracing_models.nn_models.Models as M

cnnlstm = M.CNNLSTM(context_length=10, sequence_length=1)
gpu = 1
cnnlstm = cnnlstm.cuda(gpu)
cnnlstm.eval()



im = torch.rand( 128, 10, 3, 66, 200, device="cuda:%d"% (gpu) )
print(cnnlstm)
out = cnnlstm(im)
outsqueezed = out.squeeze()
print(out.shape)
print(outsqueezed.shape)

