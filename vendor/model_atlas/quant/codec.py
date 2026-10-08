"""Reference codec, CPU/GPU compatible. Packed storage, floating point compute.
S12 v1: 0 -> zero; 1..4095 -> 2**(-32 + (code-1)*48/4094).
Fixed range needs no per-tensor anchor. Codes/scales are separate byte streams.
"""
import torch
from .vendor.base3_pack import pack_ktrit, unpack_ktrit

def encode_scale(s, bits):
    if bits == 12:
        if not torch.isfinite(s).all() or (s > 2**16).any():
            raise ValueError('S12 scale out of representable range')
        code = ((s.clamp_min(2**-32).log2()+32)*4094/48).round().clamp(0,4094).to(torch.int32)+1
        code = torch.where(s==0,0,code)
        decoded = torch.where(code==0,0.,torch.exp2(-32+(code.float()-1)*48/4094))
        return code, decoded
    q=s.to(torch.float16 if bits==16 else torch.float32)
    if not torch.isfinite(q).all(): raise ValueError('Scale overflow')
    return q,q.float()

def pack12(values):
    v=values.reshape(-1).to(torch.int32)
    if ((v<0)|(v>4095)).any(): raise ValueError('Invalid S12 code')
    n=v.numel()
    if n%2: v=torch.cat([v,v.new_zeros(1)])
    a,b=v[::2],v[1::2]
    out=torch.stack([a&255,(a>>8)|((b&15)<<4),b>>4],1).to(torch.uint8).flatten()
    return out[:(n*12+7)//8]

def unpack12(data, count):
    if data.numel() != (count*12+7)//8: raise ValueError('Bad S12 payload length')
    v=data.to(torch.int32)
    if v.numel()%3: v=torch.cat([v,v.new_zeros(3-v.numel()%3)])
    v=v.reshape(-1,3)
    return torch.stack([v[:,0]|((v[:,1]&15)<<8),(v[:,1]>>4)|(v[:,2]<<4)],1).flatten()[:count]

def pack(codes, scale_codes, spec):
    o,i=codes.shape;g=spec['group'];k=spec['k'];ng=(i+g-1)//g
    if spec['scale_bits']==12:
        c=torch.nn.functional.pad(codes,(0,ng*g-i)).reshape(o*ng,g)
    else: c=codes
    packed,_,_=pack_ktrit(c,k)
    scales=pack12(scale_codes) if spec['scale_bits']==12 else scale_codes
    return {'codes':packed.contiguous().cpu(),'scales':scales.contiguous().cpu()}

def unpack(tensors, shape, spec, device='cpu'):
    o,i=shape;g=spec['group'];k=spec['k'];ng=(i+g-1)//g
    c=unpack_ktrit(tensors['codes'].to(device),g if spec['scale_bits']==12 else i,k)
    if spec['scale_bits']==12:
        c=c.reshape(o,ng*g)[:,:i]
        sc=unpack12(tensors['scales'].to(device),o*ng).reshape(o,ng)
        s=torch.where(sc==0,0.,torch.exp2(-32+(sc.float()-1)*48/4094))
    else: s=tensors['scales'].to(device).float().reshape(o,ng)
    return c.float()/((3**k-1)//2)*s.repeat_interleave(g,1)[:,:i]

def fit_scales(w, diag, spec):
    o,i=w.shape;g=spec['group'];ng=(i+g-1)//g;L=(3**spec['k']-1)//2
    x=torch.nn.functional.pad(w,(0,ng*g-i)).reshape(o,ng,g)
    d=torch.nn.functional.pad(diag,(0,ng*g-i)).reshape(1,ng,g)
    maximum=x.abs().amax(-1); best=None
    for ratio in torch.linspace(.55,1.,10,device=w.device):
        codes,s=encode_scale(maximum*ratio,spec['scale_bits'])
        q=(x/s.clamp_min(1e-30).unsqueeze(-1)*L).round().clamp(-L,L)*s.unsqueeze(-1)/L
        err=((x-q).square()*d).sum(-1)
        if best is None: best=err; bc=codes; bs=s
        else:
            take=err<best; best=torch.minimum(best,err);bc=torch.where(take,codes,bc);bs=torch.where(take,s,bs)
    return bc,bs

@torch.no_grad()
def quantize(w, h, spec, row_chunk=128, damp=.01):
    """GPTQ inverse-Hessian column feedback with representable, prefit group scales.
    h is an input Gram Hessian; None means separable embedding row reconstruction.
    """
    o,i=w.shape;g=spec['group'];L=(3**spec['k']-1)//2
    if spec['scale_bits']!=12 and i%g: raise ValueError('Standard TKinteger requires input width divisible by group')
    if h is not None:
        h=h.to(device=w.device,dtype=torch.float32)
        if h.shape!=(i,i) or not torch.isfinite(h).all(): raise ValueError('Invalid Hessian')
        diag=h.diag().clone(); h=(h+h.T)*.5
        jitter=max(float(diag.mean())*damp,1e-8)
        for _ in range(8):
            chol,info=torch.linalg.cholesky_ex(h+torch.eye(i,device=w.device)*jitter)
            if int(info.max())==0: break
            jitter*=10
        else: raise ValueError('Hessian not positive definite after damping')
        inv=torch.cholesky_inverse(chol); u=torch.linalg.cholesky(inv,upper=True)
    else: diag=w.new_ones(i,dtype=torch.float32);u=None
    all_codes=[];all_scales=[]
    for start in range(0,o,row_chunk):
        x=w[start:start+row_chunk].float().clone()
        sc,s=fit_scales(x,diag,spec);c=torch.zeros_like(x,dtype=torch.int32)
        if u is None:
            c=(x/s.repeat_interleave(g,1)[:,:i].clamp_min(1e-30)*L).round().clamp(-L,L).int()
        else:
            for lo in range(0,i,128):
                hi=min(i,lo+128);errors=torch.zeros_like(x[:,lo:hi])
                for j in range(lo,hi):
                    scale=s[:,j//g]; qcode=(x[:,j]/scale.clamp_min(1e-30)*L).round().clamp(-L,L)
                    c[:,j]=qcode.int();q=qcode*scale/L;e=(x[:,j]-q)/u[j,j]
                    x[:,j:hi]-=e[:,None]*u[j,j:hi];errors[:,j-lo]=e
                x[:,hi:]-=errors@u[lo:hi,hi:]
        all_codes.append(c.cpu());all_scales.append(sc.cpu())
    return pack(torch.cat(all_codes),torch.cat(all_scales),spec)
