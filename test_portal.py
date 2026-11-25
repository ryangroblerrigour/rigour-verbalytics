import os,time,io
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from batch_portal import install
os.environ.update(PORTAL_USERNAME='test',PORTAL_PASSWORD='test-password-12345',PORTAL_INSECURE_LOCAL='1')
async def generate(q,r,p,l,**kw):
    assert kw['context']['P3']=='4'; return 'What makes that important to you?'
app=FastAPI();install(app,generate,lambda *a:[],lambda *a:None,'flag')
with TestClient(app) as c:
    h={'X-Portal-Request':'1'}
    assert c.post('/portal/upload',headers=h,files={'file':('a.csv',b'x\n1')}).status_code==401
    assert c.post('/portal/login',data={'username':'test','password':'test-password-12345'}).status_code==200
    d=c.post('/portal/upload',headers=h,files={'file':('a.csv',b'project_id,respid,question_id,question,response,P3\np,001,Q,Why?,No thanks,4\np,002,Q,Why?,,4')}).json(); ident=d['id']; path='/portal/jobs/'+ident
    m={k:k for k in ['project_id','respid','question_id','question','response']};m['locale']=''
    assert c.post(path+'/run',headers=h,json={'mode':'test','mapping':m,'defaults':{},'context':{'P3':'P3'}}).status_code==200
    for _ in range(100):
        s=c.get(path,headers=h).json()
        if not s['running']:break
        time.sleep(.01)
    assert s['success']==1 and s['skipped']==1,s
    w=load_workbook(io.BytesIO(c.get(path+'/download',headers=h).content));assert w.active['B2'].value=='001';assert w.active['G2'].value.endswith('?')
    assert c.delete(path,headers=h).status_code==204
print('PASS: authentication, upload, mapping, test generation, blank skip, export, deletion')
