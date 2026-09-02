#!/usr/bin/env python3
import json, math, re, secrets, statistics, threading, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
BASE='http://127.0.0.1:18080'; MODEL='GLM-5.3-Flash-EXL3-2.05'
TEMP=1.0; TOP_P=0.95; THINKING=False; MAX_TOKENS=16
TARGETS=(8000,16000,100000); PAIRS=3; PAGE=3584
OUT=Path(__file__).resolve().parent

def post(path,body,timeout=900):
 data=json.dumps(body).encode(); req=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json'},method='POST'); return urllib.request.urlopen(req,timeout=timeout)
def get(path,timeout=30):
 with urllib.request.urlopen(BASE+path,timeout=timeout) as r:return r.status,r.read().decode('utf-8','replace')
def tokenize(messages):
 with post('/tokenize',{'model':MODEL,'messages':messages},180) as r:return int(json.loads(r.read())['count'])
def metrics_raw(): return get('/metrics')[1]
def metrics(raw=None):
 raw=raw if raw is not None else metrics_raw(); o={}
 for ln in raw.splitlines():
  if not ln or ln.startswith('#'):continue
  try:k,v=ln.rsplit(None,1); v=float(v)
  except:continue
  n=k.split('{',1)[0]
  if n in ('vllm:prefix_cache_hits_total','vllm:prefix_cache_queries_total','vllm:prompt_tokens_total','vllm:request_success_total','vllm:num_requests_running','vllm:num_requests_waiting','vllm:gpu_cache_usage_perc'):
   o[n]=o.get(n,0)+v
  if n=='vllm:prompt_tokens_by_source_total':
   m=re.search(r'source="([^"]+)"',k)
   if m:o['source:'+m.group(1)]=o.get('source:'+m.group(1),0)+v
 return o
def mdiff(a,b):
 ks=set(a)|set(b); return {k:b.get(k,0)-a.get(k,0) for k in sorted(ks) if k not in ('vllm:num_requests_running','vllm:num_requests_waiting','vllm:gpu_cache_usage_perc')}
def wait_idle(timeout=120):
 end=time.time()+timeout
 while time.time()<end:
  m=metrics(); r=int(m.get('vllm:num_requests_running',0)); w=int(m.get('vllm:num_requests_waiting',0))
  if r==0 and w==0:return {'utc':time.strftime('%FT%TZ',time.gmtime()),'running':r,'waiting':w,'gpu_cache_usage':m.get('vllm:gpu_cache_usage_perc')}
  time.sleep(.25)
 raise RuntimeError(f'not idle: {r}/{w}')
def build(target,index):
 salt=f'{uuid.uuid4()}-{secrets.token_hex(12)}'; code=f'CK{index:02d}{secrets.token_hex(3).upper()}'
 prefix=f'UNIQUE SESSION {salt}. The required checksum is {code}.\n'
 suffix=f'\nAccording to this document, what is the required checksum? Reply with {code} only.'
 def msgs(n):return [{'role':'user','content':prefix+('the '*n)+suffix}]
 overhead=tokenize(msgs(0)); n=max(1,target-overhead); got=tokenize(msgs(n))
 for _ in range(3):
  if abs(got-target)<=2:break
  n=max(1,n+(target-got)); got=tokenize(msgs(n))
 return {'salt':salt,'code':code,'messages':msgs(n),'filler_count':n,'tokenize_count':got,'target':target}
def stream(messages,timeout=900,barrier=None):
 body={'model':MODEL,'messages':messages,'temperature':TEMP,'top_p':TOP_P,'max_tokens':MAX_TOKENS,'stream':True,
       'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
 if barrier:barrier.wait()
 t0=time.perf_counter(); first=None; chunks=[]; reasoning=[]; usage={}; finish=None; rid=None; http=None
 with post('/v1/chat/completions',body,timeout) as resp:
  http=resp.status
  for raw in resp:
   line=raw.strip()
   if not line.startswith(b'data:'):continue
   payload=line[5:].strip()
   if payload==b'[DONE]':continue
   try:o=json.loads(payload)
   except:continue
   rid=rid or o.get('id')
   if o.get('usage'):usage=o['usage']
   ch=o.get('choices') or []
   if not ch:continue
   d=ch[0].get('delta') or {}; text=d.get('content') or ''
   reason=d.get('reasoning') or d.get('reasoning_content') or ''
   if reason:reasoning.append(reason)
   if text:
    if first is None:first=time.perf_counter()
    chunks.append(text)
   if ch[0].get('finish_reason'):finish=ch[0]['finish_reason']
 t1=time.perf_counter(); text=''.join(chunks); pt=int(usage.get('prompt_tokens') or 0)
 return {'id':rid,'http':http,'t0':t0,'first':first,'end':t1,'ttft_s':None if first is None else first-t0,'wall_s':t1-t0,
         'prompt_tokens':pt,'completion_tokens':int(usage.get('completion_tokens') or 0),'prefill_tok_s':pt/(first-t0) if first and pt else None,
         'finish_reason':finish,'text':text,'reasoning':''.join(reasoning),'usage':usage,'nan':'nan' in (text+''.join(reasoning)).lower()}
def measured(messages,code,kind,timeout=900):
 wait_idle(); time.sleep(.15); a=metrics(); r=stream(messages,timeout); b=metrics(); r['kind']=kind; r['expected_code']=code; r['correct']=code in r['text']; r['metrics_delta']=mdiff(a,b); r['gpu_cache_usage_after']=b.get('vllm:gpu_cache_usage_perc'); r['contaminated']=r['metrics_delta'].get('vllm:request_success_total',0)!=1
 if r['contaminated']:raise RuntimeError(f'contamination in {kind}: {r["metrics_delta"]}')
 return r
def follow_messages(seed,cold):
 return seed['messages']+[{'role':'assistant','content':cold['text']},{'role':'user','content':'Repeat the same required checksum. Reply with the checksum code only.'}]
def compact(r):
 return {k:r.get(k) for k in ('kind','http','prompt_tokens','completion_tokens','ttft_s','prefill_tok_s','finish_reason','text','correct','nan','metrics_delta','gpu_cache_usage_after','contaminated')}
def main():
 status,models=get('/v1/models'); model_data=json.loads(models)['data'][0]
 meta={'schema':1,'started_utc':time.strftime('%FT%TZ',time.gmtime()),'base':BASE,'model':MODEL,'max_model_len':model_data.get('max_model_len'),
       'temperature':TEMP,'top_p':TOP_P,'thinking':False,'max_tokens':MAX_TOKENS,'targets':TARGETS,'pairs_each':PAIRS,
       'page_model_tokens':PAGE,'scope':'3 sequential unique cold/warm pairs at 8K, 16K, 100K plus 4 independently seeded concurrent 8K warm follow-ups'}
 meta['idle_before']=wait_idle(); (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 rec={'pairs':[],'concurrent_8k':None}
 idx=0
 for target in TARGETS:
  for n in range(1,PAIRS+1):
   idx+=1; seed=build(target,idx)
   cold=measured(seed['messages'],seed['code'],f'{target}-cold-{n}',timeout=900)
   warm=measured(follow_messages(seed,cold),seed['code'],f'{target}-warm-{n}',timeout=900)
   row={'target':target,'pair':n,'seed':{k:seed[k] for k in ('code','filler_count','tokenize_count','target')},'cold':cold,'warm':warm}
   rec['pairs'].append(row)
   print(json.dumps({'target':target,'pair':n,'cold':compact(cold),'warm':compact(warm)}),flush=True)
 # Four independent 8K cold seeds loaded sequentially; then warm follow-ups start together.
 prepared=[]
 for j in range(4):
  idx+=1; seed=build(8000,idx); cold=measured(seed['messages'],seed['code'],f'concurrent-seed-{j+1}-cold',timeout=300); prepared.append((seed,cold))
  print(json.dumps({'concurrent_seed':j+1,'cold':compact(cold)}),flush=True)
 wait_idle(); time.sleep(.2); before=metrics(); barrier=threading.Barrier(4)
 with ThreadPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(stream,follow_messages(s,c),300,barrier) for s,c in prepared]; warms=[f.result() for f in fs]
 after=metrics(); d=mdiff(before,after)
 for i,(r,(s,c)) in enumerate(zip(warms,prepared),1):r.update({'kind':f'concurrent-warm-{i}','expected_code':s['code'],'correct':s['code'] in r['text']})
 contam=d.get('vllm:request_success_total',0)!=4
 if contam:raise RuntimeError(f'contamination concurrent warms: {d}')
 rec['concurrent_8k']={'seeds':[{'seed':{k:s[k] for k in ('code','filler_count','tokenize_count','target')},'cold':c} for s,c in prepared],
                       'warms':warms,'metrics_delta':d,'contaminated':contam,
                       'ttft_min_s':min(r['ttft_s'] for r in warms),'ttft_max_s':max(r['ttft_s'] for r in warms),
                       'all_correct':all(r['correct'] for r in warms),'all_http_200':all(r['http']==200 for r in warms)}
 print(json.dumps({'concurrent_8k':{'warms':[compact(r) for r in warms],'metrics_delta':d,'all_correct':rec['concurrent_8k']['all_correct']}}),flush=True)
 # Summaries from observed counters, without assuming page size.
 summary={'targets':{}}
 for target in TARGETS:
  rows=[x for x in rec['pairs'] if x['target']==target]; cs=[x['cold'] for x in rows]; ws=[x['warm'] for x in rows]
  def med(key,arr):return statistics.median(x[key] for x in arr if x.get(key) is not None)
  def hit(x):return x['metrics_delta'].get('vllm:prefix_cache_hits_total',x['metrics_delta'].get('source:local_cache_hit',0))
  summary['targets'][str(target)]={'pairs':3,'median_prompt_tokens_cold':med('prompt_tokens',cs),'median_cold_ttft_s':med('ttft_s',cs),
    'median_cold_prefill_tok_s':med('prefill_tok_s',cs),'cold_hits':[hit(x) for x in cs],
    'median_warm_prompt_tokens':med('prompt_tokens',ws),'median_warm_ttft_s':med('ttft_s',ws),'warm_hits':[hit(x) for x in ws],
    'warm_hit_ratio_median':statistics.median(hit(x)/x['prompt_tokens'] for x in ws),
    'all_correct':all(x['correct'] for x in cs+ws),'all_http_200':all(x['http']==200 for x in cs+ws),'any_nan':any(x['nan'] for x in cs+ws)}
 summary['concurrent_8k']={'ttft_min_s':rec['concurrent_8k']['ttft_min_s'],'ttft_max_s':rec['concurrent_8k']['ttft_max_s'],
   'aggregate_hits':d.get('vllm:prefix_cache_hits_total',d.get('source:local_cache_hit',0)),
   'aggregate_prompt_tokens':sum(r['prompt_tokens'] for r in warms),'all_correct':rec['concurrent_8k']['all_correct'],'all_http_200':rec['concurrent_8k']['all_http_200']}
 rec['summary']=summary; (OUT/'results.json').write_text(json.dumps(rec,indent=2)+'\n')
 meta['finished_utc']=time.strftime('%FT%TZ',time.gmtime()); meta['idle_after']=wait_idle(); meta['health_after']=get('/health')[0]
 (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('FINAL '+json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
