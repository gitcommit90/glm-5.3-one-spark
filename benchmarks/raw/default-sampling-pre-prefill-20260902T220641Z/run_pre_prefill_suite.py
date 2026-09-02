#!/usr/bin/env python3
import json, math, re, statistics, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE='http://127.0.0.1:18080'
MODEL='GLM-5.3-Flash-EXL3-2.05'
TEMP=1.0
TOP_P=0.95
MAX_TOKENS=400
STRUCTURED='Count from 1 to 200. Output only the numbers, separated by spaces. No other text.'
PROSE='Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.'
OUTDIR=Path(__file__).resolve().parent
SPEC_NAMES=(
 'vllm:spec_decode_num_drafts_total',
 'vllm:spec_decode_num_draft_tokens_total',
 'vllm:spec_decode_num_accepted_tokens_total',
)

def http_get(path, timeout=20):
    with urllib.request.urlopen(BASE+path, timeout=timeout) as r:
        return r.status, r.read().decode('utf-8','replace')

def metrics_text(): return http_get('/metrics')[1]

def parse_metrics(text):
    out={}
    for line in text.splitlines():
        if not line or line.startswith('#'): continue
        try: key,val=line.rsplit(None,1); val=float(val)
        except: continue
        name=key.split('{',1)[0]
        out[name]=out.get(name,0.0)+val
        if name=='vllm:spec_decode_num_accepted_tokens_per_pos_total':
            m=re.search(r'position="(\d+)"',key)
            if m: out[f'pos:{m.group(1)}']=out.get(f'pos:{m.group(1)}',0.0)+val
    return out

def delta(before,after):
    drafts=after.get(SPEC_NAMES[0],0)-before.get(SPEC_NAMES[0],0)
    draft_tokens=after.get(SPEC_NAMES[1],0)-before.get(SPEC_NAMES[1],0)
    accepted=after.get(SPEC_NAMES[2],0)-before.get(SPEC_NAMES[2],0)
    return {
      'drafts':int(drafts),'draft_tokens':int(draft_tokens),'accepted':int(accepted),
      'accept_ratio':accepted/draft_tokens if draft_tokens else None,
      'accepted_per_step':accepted/drafts if drafts else None,
      'per_position':[ ((after.get(f'pos:{i}',0)-before.get(f'pos:{i}',0))/drafts if drafts else None) for i in range(7)]
    }

def scheduler_state(text):
    p=parse_metrics(text)
    return int(p.get('vllm:num_requests_running',0)),int(p.get('vllm:num_requests_waiting',0))

def wait_idle(timeout=60):
    end=time.time()+timeout
    while time.time()<end:
        r,w=scheduler_state(metrics_text())
        if r==0 and w==0: return {'running':r,'waiting':w,'utc':time.strftime('%FT%TZ',time.gmtime())}
        time.sleep(.25)
    raise RuntimeError(f'backend not idle after {timeout}s: running={r} waiting={w}')

def stream_one(prompt,max_tokens=MAX_TOKENS,barrier=None):
    body={'model':MODEL,'messages':[{'role':'user','content':prompt}],
          'temperature':TEMP,'top_p':TOP_P,'max_tokens':max_tokens,'stream':True,
          'stream_options':{'include_usage':True},
          'chat_template_kwargs':{'enable_thinking':False}}
    data=json.dumps(body).encode()
    req=urllib.request.Request(BASE+'/v1/chat/completions',data=data,headers={'Content-Type':'application/json'},method='POST')
    if barrier: barrier.wait()
    t0=time.perf_counter(); first=None; end=None; usage={}; finish=None; chunks=[]; rid=None; http=None
    with urllib.request.urlopen(req,timeout=900) as resp:
        http=resp.status; buf=b''
        while True:
            piece=resp.read(256)
            if not piece: break
            buf+=piece
            while b'\n' in buf:
                line,buf=buf.split(b'\n',1); line=line.strip()
                if not line.startswith(b'data:'): continue
                payload=line[5:].strip()
                if payload==b'[DONE]': continue
                try: obj=json.loads(payload)
                except: continue
                rid=rid or obj.get('id')
                if obj.get('usage'): usage=obj['usage']
                choices=obj.get('choices') or []
                if not choices: continue
                d=choices[0].get('delta') or {}
                text=d.get('content') or d.get('reasoning') or d.get('reasoning_content') or ''
                if text:
                    if first is None: first=time.perf_counter()
                    chunks.append(text)
                if choices[0].get('finish_reason'): finish=choices[0]['finish_reason']
    end=time.perf_counter(); text=''.join(chunks)
    ct=int(usage.get('completion_tokens') or 0); pt=int(usage.get('prompt_tokens') or 0)
    dec_tokens=max(ct-1,0); dec_s=(end-first) if first is not None else None
    return {'id':rid,'http':http,'t0':t0,'first':first,'end':end,'ttft_s':None if first is None else first-t0,
            'wall_s':end-t0,'decode_s':dec_s,'decode_tokens':dec_tokens,'tok_s':dec_tokens/dec_s if dec_s else None,
            'prompt_tokens':pt,'completion_tokens':ct,'finish_reason':finish,
            'text_head':text[:400],'text_len':len(text),'nan':('nan' in text.lower() or 'locklock' in text.lower())}

def sampler(stop,rec):
    while not stop.is_set():
        try:
            r,w=scheduler_state(metrics_text()); rec.append({'t':time.time(),'running':r,'waiting':w})
        except Exception as e: rec.append({'t':time.time(),'error':str(e)})
        stop.wait(.05)

def run_concurrent(c):
    idle=wait_idle(); time.sleep(.3)
    before_txt=metrics_text(); before=parse_metrics(before_txt)
    barrier=threading.Barrier(c); stop=threading.Event(); samples=[]
    th=threading.Thread(target=sampler,args=(stop,samples),daemon=True); th.start()
    with ThreadPoolExecutor(max_workers=c) as ex:
        futs=[ex.submit(stream_one,STRUCTURED,MAX_TOKENS,barrier) for _ in range(c)]
        runs=[f.result() for f in futs]
    stop.set(); th.join(2)
    after_txt=metrics_text(); after=parse_metrics(after_txt); spec=delta(before,after)
    success_delta=after.get('vllm:request_success_total',0)-before.get('vllm:request_success_total',0)
    earliest=min(r['first'] for r in runs if r['first'] is not None); latest=max(r['end'] for r in runs)
    agg_tokens=sum(r['decode_tokens'] for r in runs)
    return {'concurrency':c,'idle_before':idle,'streams':runs,'spec':spec,
            'aggregate_decode_tok_s':agg_tokens/(latest-earliest),'aggregate_decode_tokens':agg_tokens,
            'aggregate_decode_window_s':latest-earliest,
            'median_stream_tok_s':statistics.median(r['tok_s'] for r in runs),
            'median_ttft_s':statistics.median(r['ttft_s'] for r in runs),
            'peak_running':max((x.get('running',0) for x in samples),default=0),
            'peak_waiting':max((x.get('waiting',0) for x in samples),default=0),
            'request_success_delta':success_delta,'contaminated':success_delta!=c,
            'samples':samples}

def median(xs): return statistics.median([x for x in xs if x is not None])

def main():
    code,health=http_get('/health'); idle=wait_idle()
    meta={'schema':1,'started_utc':time.strftime('%FT%TZ',time.gmtime()),'base':BASE,'model':MODEL,
          'temperature':TEMP,'top_p':TOP_P,'thinking':False,'dflash_k':7,'max_tokens':MAX_TOKENS,
          'health':code,'idle':idle,'structured_prompt':STRUCTURED,'prose_prompt':PROSE}
    (OUTDIR/'metadata.json').write_text(json.dumps(meta,indent=2))
    # Exactly one excluded 32-token warmup for structured concurrency suite.
    warm=stream_one(STRUCTURED,32)
    conc={'warmup':warm,'rounds':[]}
    for c in (1,2,4):
        for n in range(1,6):
            rr=run_concurrent(c); rr['round']=n; conc['rounds'].append(rr)
            print(json.dumps({'suite':'concurrency','c':c,'round':n,'stream_tps':[round(x['tok_s'],3) for x in rr['streams']],
                              'median_stream':round(rr['median_stream_tok_s'],3),'aggregate':round(rr['aggregate_decode_tok_s'],3),
                              'accept':round(rr['spec']['accept_ratio'],4),'accepted_per_step':round(rr['spec']['accepted_per_step'],3),
                              'peak_running':rr['peak_running'],'peak_waiting':rr['peak_waiting'],'contaminated':rr['contaminated']}),flush=True)
            if rr['contaminated']: raise RuntimeError(f'external successful request contaminated C{c} round {n}')
    conc['summary']={}
    for c in (1,2,4):
        rs=[r for r in conc['rounds'] if r['concurrency']==c]
        conc['summary'][f'C{c}']={
          'rounds':len(rs),'median_stream_tok_s':median([r['median_stream_tok_s'] for r in rs]),
          'median_aggregate_tok_s':median([r['aggregate_decode_tok_s'] for r in rs]),
          'median_ttft_s':median([r['median_ttft_s'] for r in rs]),
          'median_accept_ratio':median([r['spec']['accept_ratio'] for r in rs]),
          'median_accepted_per_step':median([r['spec']['accepted_per_step'] for r in rs]),
          'peak_running':max(r['peak_running'] for r in rs),'peak_waiting':max(r['peak_waiting'] for r in rs),
          'all_http_200':all(s['http']==200 for r in rs for s in r['streams']),
          'all_400_tokens':all(s['completion_tokens']==400 for r in rs for s in r['streams']),
          'any_nan':any(s['nan'] for r in rs for s in r['streams'])}
    (OUTDIR/'concurrency.json').write_text(json.dumps(conc,indent=2))
    # Prose: five sequential measured requests, no added prompt probes.
    prose={'runs':[]}
    for n in range(1,6):
        wait_idle(); time.sleep(.3); before=parse_metrics(metrics_text()); r=stream_one(PROSE,400); after=parse_metrics(metrics_text())
        r['spec']=delta(before,after); r['request_success_delta']=after.get('vllm:request_success_total',0)-before.get('vllm:request_success_total',0)
        r['contaminated']=r['request_success_delta']!=1; prose['runs'].append(r)
        print(json.dumps({'suite':'prose','run':n,'tok_s':round(r['tok_s'],3),'ttft_s':round(r['ttft_s'],3),
                          'accept':round(r['spec']['accept_ratio'],4),'accepted_per_step':round(r['spec']['accepted_per_step'],3),
                          'tokens':r['completion_tokens'],'contaminated':r['contaminated']}),flush=True)
        if r['contaminated']: raise RuntimeError(f'external successful request contaminated prose run {n}')
    prose['summary']={'runs':5,'median_tok_s':median([r['tok_s'] for r in prose['runs']]),
      'median_ttft_s':median([r['ttft_s'] for r in prose['runs']]),
      'median_accept_ratio':median([r['spec']['accept_ratio'] for r in prose['runs']]),
      'median_accepted_per_step':median([r['spec']['accepted_per_step'] for r in prose['runs']]),
      'per_position_median':[median([r['spec']['per_position'][i] for r in prose['runs']]) for i in range(7)],
      'all_http_200':all(r['http']==200 for r in prose['runs']),'all_400_tokens':all(r['completion_tokens']==400 for r in prose['runs']),
      'any_nan':any(r['nan'] for r in prose['runs'])}
    (OUTDIR/'prose.json').write_text(json.dumps(prose,indent=2))
    meta['finished_utc']=time.strftime('%FT%TZ',time.gmtime()); meta['health_after']=http_get('/health')[0]; meta['idle_after']=wait_idle()
    (OUTDIR/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('FINAL',json.dumps({'concurrency':conc['summary'],'prose':prose['summary']},indent=2),flush=True)
if __name__=='__main__': main()
