"""Internal batch portal. Jobs are deliberately ephemeral and single-process."""
import asyncio, csv, io, os, secrets, time, hmac, hashlib, json, zipfile
from pathlib import Path
from fastapi import Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from openpyxl import Workbook, load_workbook


def install(app, generate, blocked, validate, reject_mode):
    jobs = {}
    revoked = {}
    limit = int(os.getenv('PORTAL_MAX_ROWS', '5000'))
    ttl = min(86400, int(os.getenv('PORTAL_RETENTION_SECONDS', '86400')))
    semaphore = asyncio.Semaphore(max(1, int(os.getenv('PORTAL_CONCURRENCY', '5'))))
    key = secrets.token_bytes(32)
    page = Path(__file__).with_name('portal.html').read_text()
    def auth(request):
        token = request.cookies.get('verbalytics_portal', '')
        try:
            expiry, nonce, signature = token.split('.')
            expected = hmac.new(key, f'{expiry}.{nonce}'.encode(), hashlib.sha256).hexdigest()
            if int(expiry) > time.time() and nonce not in revoked and hmac.compare_digest(signature, expected): return nonce
        except ValueError: pass
        raise HTTPException(401, 'Sign in again')
    def guard(request):
        owner = auth(request)
        if request.headers.get('x-portal-request') != '1': raise HTTPException(403)
        return owner
    def job(request, ident):
        owner = guard(request)
        j = jobs.get(ident)
        if not j or j['owner'] != owner or j['expires'] <= time.time(): raise HTTPException(404, 'Job expired or unavailable')
        return j
    async def cleanup():
        while True:
            await asyncio.sleep(30)
            for nonce, expiry in list(revoked.items()):
                if expiry <= time.time(): revoked.pop(nonce, None)
            for ident,j in list(jobs.items()):
                if j['expires'] <= time.time():
                    if j.get('task'): j['task'].cancel()
                    jobs.pop(ident,None)
    @app.on_event('startup')
    async def start_cleanup(): app.state.portal_cleanup = asyncio.create_task(cleanup())
    @app.on_event('shutdown')
    async def stop_cleanup():
        app.state.portal_cleanup.cancel()
        for j in jobs.values():
            if j.get('task'): j['task'].cancel()
    @app.get('/portal', response_class=HTMLResponse)
    @app.get('/portal/batch', response_class=HTMLResponse)
    async def portal(request: Request):
        try: auth(request)
        except HTTPException: return RedirectResponse('/portal/login',303)
        return HTMLResponse(page, headers={'Cache-Control':'no-store'})
    @app.get('/portal/login', response_class=HTMLResponse)
    async def login_page():
        return HTMLResponse('<meta name="viewport" content="width=device-width"><title>Verbalytics login</title><h1>Verbalytics Portal</h1><form method="post"><label>Username <input name="username" required></label><p><label>Password <input type="password" name="password" required></label></p><button>Sign in</button></form>',headers={'Cache-Control':'no-store'})
    attempts = {}
    @app.post('/portal/login')
    async def login(request: Request, username: str=Form(...), password: str=Form(...)):
        u,p = os.getenv('PORTAL_USERNAME'),os.getenv('PORTAL_PASSWORD')
        if not u or not p or len(p)<16: raise HTTPException(503,'Portal credentials not configured (password minimum 16 characters)')
        origin = request.headers.get('origin')
        if origin and origin.split('://')[-1] != request.headers.get('host'): raise HTTPException(403)
        now=time.time(); ip=request.client.host if request.client else 'unknown'
        for k,v in list(attempts.items()):
            if now-v[0]>300: attempts.pop(k,None)
        stamp,count=attempts.get(ip,(now,0))
        if count>=10: raise HTTPException(429,'Try again later')
        if not (secrets.compare_digest(username.encode(),u.encode()) & secrets.compare_digest(password.encode(),p.encode())):
            attempts[ip]=(stamp,count+1); raise HTTPException(401,'Invalid credentials')
        attempts.pop(ip,None)
        nonce=secrets.token_hex(24); value=f'{int(now)+86400}.{nonce}'
        signature=hmac.new(key,value.encode(),hashlib.sha256).hexdigest()
        r=RedirectResponse('/portal',303); r.set_cookie('verbalytics_portal',value+'.'+signature,httponly=True,secure=os.getenv('PORTAL_INSECURE_LOCAL')!='1',samesite='strict',max_age=86400,path='/portal'); return r
    @app.post('/portal/logout')
    async def logout(request: Request):
        owner=guard(request)
        revoked[owner]=time.time()+86400
        for ident,j in list(jobs.items()):
            if j['owner']==owner:
                if j.get('task'): j['task'].cancel()
                jobs.pop(ident,None)
        r=Response(status_code=204); r.delete_cookie('verbalytics_portal',path='/portal'); return r
    def parse(data,name):
        if name.lower().endswith('.csv'):
            rows=list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
        elif name.lower().endswith('.xlsx'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if sum(i.file_size for i in z.infolist())>80_000_000: raise ValueError('Expanded workbook too large')
            wb=load_workbook(io.BytesIO(data),read_only=True,data_only=True)
            try:
                if len(wb.sheetnames)!=1: raise ValueError('Upload a single-sheet workbook')
                ws=wb.active
                if ws.max_row and ws.max_row>limit+1 or ws.max_column and ws.max_column>200: raise ValueError('Workbook exceeds limits')
                rows=list(ws.values)
            finally: wb.close()
        else: raise ValueError('Use CSV or XLSX')
        if not rows or len(rows)<2 or len(rows)>limit+1: raise ValueError(f'Expected 1–{limit} data rows')
        headers=[str(x or '').strip() for x in rows[0]]
        if len(headers)>200 or any(not x for x in headers) or len(set(headers))!=len(headers): raise ValueError('Headers must be unique and nonblank (maximum 200)')
        if any(len(r)>len(headers) for r in rows[1:]): raise ValueError('Rows exceed header width')
        return headers,[list(r)+[None]*(len(headers)-len(r)) for r in rows[1:]]
    @app.post('/portal/upload')
    async def upload(request: Request, file: UploadFile=File(...)):
        owner=guard(request)
        if len(jobs)>=10 or sum(j['owner']==owner for j in jobs.values())>=3: raise HTTPException(429,'Too many jobs; delete an existing job')
        data=await file.read(10_000_001); await file.close()
        if len(data)>10_000_000: raise HTTPException(413,'Maximum upload 10 MB')
        try: headers,rows=await asyncio.to_thread(parse,data,file.filename or '')
        except Exception as e: raise HTTPException(422,str(e)[:200])
        ident=secrets.token_hex(16); jobs[ident]={'owner':owner,'headers':headers,'rows':rows,'results':[None]*len(rows),'expires':time.time()+ttl,'running':False}
        return {'id':ident,'headers':headers,'count':len(rows),'preview':rows[:5]}
    def payload(j,row,mapping,defaults,context_map):
        d=dict(zip(j['headers'],row)); fields={k:d.get(v) if v else defaults.get(k) for k,v in mapping.items()}
        for k in ('project_id','respid','question_id','question','response','locale'):
            fields[k]=str(fields.get(k) or '').strip()
        if not fields['response']: return None
        for k in ('project_id','respid','question_id','question'):
            if not fields[k]: raise ValueError('Missing '+k)
        ctx={k:d[v] for k,v in context_map.items() if d[v] is not None and d[v]!=''}
        ctx={k:(int(v) if isinstance(v,float) and v.is_integer() else v) for k,v in ctx.items()}
        ctx['question_id']=fields['question_id']; fields['context']=ctx; fields['locale']=fields['locale'] or 'en'
        validate(fields['project_id'],ctx)
        return fields
    async def run(j,indices):
        try:
            queue=asyncio.Queue()
            for i in indices: queue.put_nowait(i)
            async def worker():
                while not queue.empty():
                    i=queue.get_nowait(); f=j['payloads'][i]
                    for attempt in range(3):
                        try:
                            async with semaphore:
                                hits=blocked(f['question'],f['project_id'])+blocked(f['response'],f['project_id'])
                                if hits and reject_mode=='reject': raise HTTPException(422,'Blocked input')
                                result=await generate(f['question'],f['response'],f['project_id'],f['locale'],context=f['context'],batch_context=True)
                                if not result or blocked(result,f['project_id']): raise HTTPException(422,'Empty or blocked follow-up')
                            j['results'][i]=[result,'success','']; break
                        except Exception as e:
                            status=getattr(e,'status_code',500)
                            if attempt==2 or status in (400,401,403,422):
                                j['results'][i]=['','failed',f'Generation failed ({status})']; break
                            await asyncio.sleep(2**attempt)
            await asyncio.gather(*(worker() for _ in range(min(5,len(indices)))))
        finally: j['running']=False
    @app.post('/portal/jobs/{ident}/run')
    async def launch(request: Request,ident: str):
        j=job(request,ident); body=await request.json(); mode=body.get('mode')
        if j['running']: raise HTTPException(409,'Job already running')
        if mode not in ('test','all','retry'): raise HTTPException(422,'Invalid mode')
        if 'payloads' not in j:
            mapping=body.get('mapping',{}); defaults=body.get('defaults',{}); ctx=body.get('context',{})
            keys=('project_id','respid','question_id','question','response','locale')
            if set(mapping)!=set(keys) or any(v and v not in j['headers'] for v in mapping.values()) or any(v not in j['headers'] for v in ctx.values()): raise HTTPException(422,'Invalid mapping')
            payloads=[]; errors=[]
            for i,row in enumerate(j['rows']):
                try: payloads.append(payload(j,row,mapping,defaults,ctx))
                except Exception as e:
                    payloads.append(None); errors.append(f'Row {i+2}: {getattr(e,"detail",str(e))}')
            if errors: raise HTTPException(422,{'invalid_rows':len(errors),'examples':errors[:10]})
            j['payloads']=payloads
            for i,f in enumerate(payloads):
                if f is None: j['results'][i]=['','skipped','Blank response']
        indices=[i for i,f in enumerate(j['payloads']) if f and (j['results'][i] is None if mode!='retry' else j['results'][i] is not None and j['results'][i][1]=='failed')]
        if mode=='test': indices=[i for i in indices if i<5]
        j['running']=True; j['task']=asyncio.create_task(run(j,indices)); return {'queued':len(indices)}
    @app.get('/portal/jobs/{ident}')
    async def status(request: Request,ident: str):
        j=job(request,ident); r=j['results']; return {'running':j['running'],'total':len(r),'pending':sum(x is None for x in r),**{s:sum(x is not None and x[1]==s for x in r) for s in ('success','failed','skipped')},'preview':[[i+2,*x] for i,x in enumerate(r[:5]) if x]}
    @app.get('/portal/jobs/{ident}/download')
    async def download(request: Request,ident: str):
        j=job(request,ident)
        def export():
            wb=Workbook(); ws=wb.active; ws.title='Follow-ups'
            names=list(j['headers']); extra=[]
            for base in ('AI_followup','AI_status','AI_error'):
                name=base; n=2
                while name in names+extra: name=f'{base}_{n}'; n+=1
                extra.append(name)
            for row in [names+extra]+[r+(x or ['','pending','']) for r,x in zip(j['rows'],j['results'])]:
                ws.append(row)
                for c in ws[ws.max_row]:
                    if isinstance(c.value,str): c.data_type='s'
            ws.freeze_panes='A2'; out=io.BytesIO(); wb.save(out); return out.getvalue()
        return Response(await asyncio.to_thread(export),media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',headers={'Content-Disposition':'attachment; filename=verbalytics-results.xlsx','Cache-Control':'no-store'})
    @app.delete('/portal/jobs/{ident}')
    async def delete(request: Request,ident: str):
        j=job(request,ident)
        if j.get('task'): j['task'].cancel()
        jobs.pop(ident,None); return Response(status_code=204)
