import torch, types
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config, Exl3MoEMethod, apply_exl3_experts,
)

def layer_mul1(device, n_exp=3, hidden=256, inter=256):
    moe=types.SimpleNamespace(swiglu_limit=10.0)
    method=Exl3MoEMethod(moe, Exl3Config(bits=2.05, codebook="mul1"))
    layer=torch.nn.Module()
    method.create_weights(layer, num_experts=n_exp, hidden_size=hidden,
                          intermediate_size_per_partition=inter,
                          params_dtype=torch.float16)
    g=torch.Generator(device="cpu"); g.manual_seed(20260902)
    with torch.no_grad():
        layer.w13_trellis.copy_(torch.randint(-30000,30000,tuple(layer.w13_trellis.shape),dtype=torch.int16,generator=g))
        layer.w2_trellis.copy_(torch.randint(-30000,30000,tuple(layer.w2_trellis.shape),dtype=torch.int16,generator=g))
        layer.w13_suh.copy_(torch.randn(tuple(layer.w13_suh.shape),generator=g).half())
        layer.w13_svh.copy_(torch.randn(tuple(layer.w13_svh.shape),generator=g).half())
        layer.w2_suh.copy_(torch.randn(tuple(layer.w2_suh.shape),generator=g).half())
        layer.w2_svh.copy_(torch.randn(tuple(layer.w2_svh.shape),generator=g).half())
        layer.w13_suh[:,1].copy_(layer.w13_suh[:,0])
        layer.w13_mul1.fill_(-2082680531)
        layer.w2_mul1.fill_(-2082680531)
    layer=layer.to(device); method.process_weights_after_loading(layer)
    return layer

def main():
    device=torch.device('cuda:0'); torch.manual_seed(44)
    layer=layer_mul1(device)
    assert layer._exl3_codebook_flags == (False,True,False,True,False,True), layer._exl3_codebook_flags
    x=torch.randn(7,256,dtype=torch.float16,device=device)
    ids=torch.tensor([[0,2],[0,1],[1,2],[2,0],[1,0],[2,1],[0,2]],dtype=torch.long,device=device)
    w=torch.tensor([[.6,.4],[.5,.5],[.7,.3],[.4,.6],[.8,.2],[.55,.45],[.65,.35]],dtype=torch.float16,device=device)
    yl=apply_exl3_experts(x,ids,w,layer,fused=False)
    yf=apply_exl3_experts(x,ids,w,layer,fused=True)
    err=(yl.float()-yf.float()).abs(); max_err=float(err.max())
    bound=max(.15,.08*float(yl.float().abs().max().clamp_min(1.0)))
    print({'flags':layer._exl3_codebook_flags,'K':layer._exl3_k,'max_err':max_err,'bound':bound,'loop_finite':bool(torch.isfinite(yl).all()),'fused_finite':bool(torch.isfinite(yf).all())})
    assert torch.isfinite(yl).all() and torch.isfinite(yf).all()
    assert max_err < bound, (max_err,bound)
if __name__=='__main__': main()
