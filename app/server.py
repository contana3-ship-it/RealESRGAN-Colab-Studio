import os, re, json, time, uuid, shutil, threading, subprocess, secrets
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security.utils import get_authorization_scheme_param
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from .model import Upscaler

ROOT = Path(os.environ.get('STUDIO_DATA', '/content/realesrgan-studio-data')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
PASSWORD = os.environ.get('STUDIO_PASSWORD', '')
if not PASSWORD: raise RuntimeError('Cần STUDIO_PASSWORD')
STATUS_LOCK = threading.RLock()
WORKER_LOCK = threading.Lock()
MAX_UPLOAD_GB = float(os.environ.get('STUDIO_MAX_UPLOAD_GB','5'))

def load(jobid):
    p=ROOT/jobid/'status.json'
    if not p.exists(): raise HTTPException(404,'Không tìm thấy tác vụ')
    with STATUS_LOCK: return json.loads(p.read_text('utf-8'))

def save(j):
    with STATUS_LOCK:
        p=ROOT/j['id']/'status.json'; tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)

def update(j,**fields):
    j.update(fields); j['updated']=time.time(); save(j)

def ffprobe(path):
    cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',str(path)]
    v=json.loads(subprocess.check_output(cmd,text=True))['streams'][0]
    num,den=(v.get('avg_frame_rate') or '30/1').split('/')
    fps=float(num)/max(float(den),1)
    if fps<=0: fps=30.
    return fps,int(v['width']),int(v['height']),int(v['nb_frames']) if v.get('nb_frames','N/A').isdigit() else 0

def process(jobid):
    with WORKER_LOCK:
        j=load(jobid)
        if j['state']=='done': return
        folder=ROOT/jobid; frames=folder/'frames'; frames.mkdir(exist_ok=True)
        src=folder/j['source']; output=folder/'restored.mp4'
        try:
            update(j,state='preparing',message='Đang kiểm tra video và khởi tạo mô hình')
            import cv2
            fps,w,h,total=ffprobe(src)
            if w*4>8192 or h*4>8192: raise RuntimeError('Kích thước video x4 vượt 8192 pixel mỗi chiều; hãy dùng video đầu vào nhỏ hơn')
            j.update(fps=fps,width=w,height=h,total=total); save(j)
            import torch
            from urllib.request import urlretrieve
            weights=ROOT/'weights'/'RealESRGAN_x4plus.pth'; weights.parent.mkdir(exist_ok=True)
            if not weights.exists():
                tmp=weights.with_suffix('.download')
                from .model import WEIGHTS_URL
                urlretrieve(WEIGHTS_URL,tmp)
                tmp.rename(weights)
            model=Upscaler(weights,tile=j['tile'])
            cap=cv2.VideoCapture(str(src))
            if not cap.isOpened(): raise RuntimeError('Không đọc được video')
            update(j,state='running',message=f'Đang xử lý bằng {model.device}')
            started=time.time(); index=0
            while True:
                if (folder/'stop.flag').exists():
                    cap.release(); update(j,state='stopped',message='Đã dừng, dữ liệu được giữ để tiếp tục'); return
                ok,frame=cap.read()
                if not ok: break
                destination=frames/f'{index:08d}.jpg'
                if not destination.exists() or destination.stat().st_size==0:
                    enhanced=model.enhance(frame)
                    if not cv2.imwrite(str(destination),enhanced,[cv2.IMWRITE_JPEG_QUALITY,j['quality']]):
                        raise RuntimeError(f'Không ghi được frame {index}')
                index+=1
                if index%3==0 or (total and index==total):
                    elapsed=max(time.time()-started,.001)
                    processed=max(0,index-j.get('resume_from',0)); rate=processed/elapsed
                    remaining=(total-index)/rate if total and rate else None
                    update(j,completed=index,progress=round(index/max(total,index)*100,2) if total else 0,
                           speed=round(rate,2),eta=round(remaining) if remaining is not None else None,
                           message=f'Đã xử lý {index} khung hình')
            cap.release()
            if index==0: raise RuntimeError('Video không có khung hình')
            update(j,state='encoding',completed=index,total=index,progress=100,message='Đang ghép video và giữ âm thanh gốc')
            # Frame sequence and source audio, if any. Use a temporary output then atomic move.
            temp=folder/'restored.partial.mp4'
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(frames/'%08d.jpg'),'-i',str(src),
                 '-map','0:v:0','-map','1:a:0?','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                 '-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',str(temp)]
            p=subprocess.run(cmd,capture_output=True,text=True)
            if p.returncode: raise RuntimeError('FFmpeg: '+p.stderr[-1200:])
            temp.replace(output)
            if j.get('clean_frames',True): shutil.rmtree(frames,ignore_errors=True)
            update(j,state='done',message='Hoàn thành',output='restored.mp4',eta=0,progress=100)
        except Exception as e:
            update(j,state='error',message=str(e)[:1600])

def kick(jobid):
    if WORKER_LOCK.locked(): raise HTTPException(409,'Đang có tác vụ khác xử lý; hãy chờ tác vụ đó kết thúc')
    threading.Thread(target=process,args=(jobid,),daemon=True).start()

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        path=request.url.path
        if path=='/health': return await call_next(request)
        scheme, token=get_authorization_scheme_param(request.headers.get('Authorization',''))
        import base64
        try:
            userpass=base64.b64decode(token).decode('utf-8') if scheme.lower()=='basic' else ''
            username,password=userpass.split(':',1)
        except Exception: username,password='',''
        valid=secrets.compare_digest(username,'admin') and secrets.compare_digest(password,PASSWORD)
        if not valid:
            return Response('Cần đăng nhập',status_code=401,headers={'WWW-Authenticate':'Basic realm="RealESRGAN Studio"'})
        return await call_next(request)

app=FastAPI(title='RealESRGAN Colab Studio')
app.add_middleware(AuthMiddleware)
app.mount('/static',StaticFiles(directory=str(Path(__file__).parent/'static')),name='static')

@app.get('/')
def index(): return FileResponse(Path(__file__).parent/'static'/'index.html')

@app.get('/health')
def health(): return {'ok':True}

@app.get('/api/jobs')
def jobs():
    entries=[]
    for p in ROOT.glob('*/status.json'):
        try: entries.append(json.loads(p.read_text('utf-8')))
        except Exception: pass
    return sorted(entries,key=lambda x:x.get('created',0),reverse=True)[:100]

@app.post('/api/jobs')
async def upload(file:UploadFile=File(...),tile:int=Form(128),quality:int=Form(95),clean_frames:bool=Form(True)):
    if WORKER_LOCK.locked(): raise HTTPException(409,'Hệ thống đang xử lý video khác')
    if not 32<=tile<=1024 or not 65<=quality<=100: raise HTTPException(400,'Tham số không hợp lệ')
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in {'.mp4','.mov','.mkv','.avi','.webm'}: raise HTTPException(400,'Chỉ hỗ trợ định dạng video')
    jid=uuid.uuid4().hex[:12]; folder=ROOT/jid; folder.mkdir()
    target=folder/('input'+suffix); size=0
    try:
        with target.open('wb') as out:
            while data:=await file.read(4*1024*1024):
                size+=len(data)
                if size>MAX_UPLOAD_GB*1024**3: raise HTTPException(413,'Video vượt giới hạn tải lên')
                out.write(data)
    except Exception:
        shutil.rmtree(folder,ignore_errors=True); raise
    j={'id':jid,'name':Path(file.filename or 'video').name,'source':target.name,'state':'queued',
       'created':time.time(),'updated':time.time(),'completed':0,'total':0,'progress':0,'speed':0,'eta':None,
       'message':'Đang xếp hàng','tile':tile,'quality':quality,'clean_frames':clean_frames,'resume_from':0}
    save(j)
    kick(jid)
    return j

@app.post('/api/upload/init')
async def init_upload(filename: str=Form(...), total_bytes:int=Form(...), tile:int=Form(128), quality:int=Form(95), clean_frames:bool=Form(True)):
    if WORKER_LOCK.locked(): raise HTTPException(409,'Hệ thống đang chạy tác vụ khác')
    if total_bytes<=0 or total_bytes>MAX_UPLOAD_GB*1024**3: raise HTTPException(413,'Kích thước video không hợp lệ')
    if not 32<=tile<=1024 or not 65<=quality<=100: raise HTTPException(400,'Tham số không hợp lệ')
    suffix=Path(filename).suffix.lower()
    if suffix not in {'.mp4','.mov','.mkv','.avi','.webm'}: raise HTTPException(400,'Chỉ hỗ trợ video')
    jid=uuid.uuid4().hex[:12]; folder=ROOT/jid; folder.mkdir()
    j={'id':jid,'name':Path(filename).name,'source':'input'+suffix,'state':'uploading',
       'created':time.time(),'updated':time.time(),'completed':0,'total':0,'progress':0,'speed':0,'eta':None,
       'message':'Đang tải video theo từng phần','tile':tile,'quality':quality,
       'clean_frames':clean_frames,'resume_from':0,'uploaded_bytes':0,'total_bytes':total_bytes}
    save(j); return j

@app.post('/api/upload/{jobid}/chunk')
async def upload_chunk(jobid:str, request:Request):
    j=load(jobid)
    if j['state']!='uploading': raise HTTPException(409,'Upload không còn hoạt động')
    expected=int(request.headers.get('X-Offset','-1'))
    target=ROOT/jobid/j['source']; actual=target.stat().st_size if target.exists() else 0
    if expected!=actual: raise HTTPException(409,f'Offset không hợp lệ; cần {actual}')
    data=await request.body()
    if not data or len(data)>16*1024*1024: raise HTTPException(413,'Chunk vượt 16 MB')
    if actual+len(data)>j['total_bytes']: raise HTTPException(413,'Chunk vượt kích thước video khai báo')
    with target.open('ab') as out: out.write(data)
    update(j,uploaded_bytes=actual+len(data)); return {'offset':actual+len(data)}

@app.post('/api/upload/{jobid}/finish')
def finish_upload(jobid:str):
    j=load(jobid)
    if j['state']!='uploading': raise HTTPException(409,'Đã hoàn tất upload')
    if not (ROOT/jobid/j['source']).exists() or (ROOT/jobid/j['source']).stat().st_size!=j.get('total_bytes'): raise HTTPException(400,'Video tải lên chưa đủ dữ liệu')
    update(j,state='queued',message='Đang chờ GPU')
    kick(jobid)
    return j

@app.get('/api/jobs/{jobid}')
def get_job(jobid:str): return load(jobid)

@app.post('/api/jobs/{jobid}/stop')
def stop(jobid:str):
    j=load(jobid)
    if j['state'] not in ('running','preparing'): raise HTTPException(409,'Không thể dừng ở trạng thái hiện tại')
    (ROOT/jobid/'stop.flag').touch()
    return {'ok':True,'message':'Đã yêu cầu dừng; sẽ dừng sau frame hiện tại'}

@app.post('/api/jobs/{jobid}/resume')
def resume(jobid:str):
    j=load(jobid)
    if j['state'] not in ('stopped','error'): raise HTTPException(409,'Tác vụ không ở trạng thái tiếp tục được')
    (ROOT/jobid/'stop.flag').unlink(missing_ok=True)
    j['resume_from']=j.get('completed',0)
    update(j,state='queued',message='Đang tiếp tục')
    kick(jobid)
    return j

@app.get('/api/jobs/{jobid}/download')
def download(jobid:str):
    j=load(jobid)
    if j['state']!='done': raise HTTPException(409,'Chưa hoàn thành')
    return FileResponse(ROOT/jobid/'restored.mp4',media_type='video/mp4',filename=f"{Path(j['name']).stem}_x4.mp4")

# Reclassify interrupted sessions so they can be resumed after Colab reconnects.
for _p in ROOT.glob('*/status.json'):
    try:
        _j=json.loads(_p.read_text('utf-8'))
        if _j.get('state') in ('queued','preparing','running','encoding'):
            update(_j,state='stopped',message='Phiên trước bị ngắt; nhấn Tiếp tục để phục hồi')
    except Exception:
        pass
