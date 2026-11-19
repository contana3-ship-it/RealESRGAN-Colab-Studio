"""Colab-native chunk uploader using the Colab notebook kernel callback bridge.
No HTTP server, Cloudflare Tunnel, or files.upload() widget required.
"""
import base64
import json
import os
import re
import uuid
from pathlib import Path

CHUNK_MAX = 2 * 1024 * 1024
ALLOWED = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}


def upload_video(destination_root: Path):
    from IPython.display import display, HTML
    from google.colab import output

    destination_root = Path(destination_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex
    folder = destination_root / session
    folder.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {'path': None, 'size': 0, 'written': 0, 'done': False}
    prefix = f'realesrgan_{session}'

    def initialize(filename, size):
        if state['path'] is not None:
            raise RuntimeError('Phiên upload đã bắt đầu')
        filename = Path(str(filename).replace('\\', '/')).name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED:
            raise ValueError('Định dạng video không được hỗ trợ')
        size = int(size)
        if size <= 0:
            raise ValueError('File rỗng')
        safe = re.sub(r'[^A-Za-z0-9._-]', '_', Path(filename).stem)[:70] or 'video'
        target = folder / f'{safe}{suffix}'
        target.touch(exist_ok=False)
        state.update(path=target, size=size)
        return {'ok': True}

    def chunk(offset, encoded):
        if state['path'] is None or state['done']:
            raise RuntimeError('Phiên upload chưa hợp lệ')
        offset = int(offset)
        if offset != state['written']:
            raise ValueError(f'Offset không khớp: nhận {offset}, mong đợi {state["written"]}')
        # FileReader.readAsDataURL returns 'data:...;base64,...'.
        payload = str(encoded).split(',', 1)[-1]
        raw = base64.b64decode(payload, validate=True)
        if not raw or len(raw) > CHUNK_MAX or offset + len(raw) > state['size']:
            raise ValueError('Kích thước chunk không hợp lệ')
        with state['path'].open('ab') as f:
            f.write(raw)
        state['written'] += len(raw)
        return {'received': state['written']}

    def finish():
        if state['written'] != state['size']:
            raise RuntimeError('Video chưa tải đầy đủ')
        if state['path'].stat().st_size != state['size']:
            raise RuntimeError('Kích thước file trên ổ đĩa không khớp')
        state['done'] = True
        return {'ok': True}

    output.register_callback(prefix + '.init', initialize)
    output.register_callback(prefix + '.chunk', chunk)
    output.register_callback(prefix + '.finish', finish)

    code = r'''(async () => {
      const id = __SESSION__, chunkSize = __CHUNK__;
      const root = document.createElement('div');
      root.style.cssText = 'font:14px system-ui;background:#142338;color:#eaf1ff;padding:20px;border-radius:14px;margin:12px 0;max-width:780px';
      root.innerHTML = `<h3 style="margin:0 0 14px">Tải video lên Colab</h3>
        <input type="file" accept="video/*,.mp4,.mov,.mkv,.avi,.webm" style="max-width:100%;color:#fff" />
        <div class="status" style="margin-top:15px">Chọn một video để tự động tải lên và khôi phục.</div>
        <div style="background:#31415b;border-radius:10px;height:15px;margin-top:10px;overflow:hidden"><div class="progress" style="height:100%;background:#35acdf;width:0%;transition:width .12s"></div></div>
        <div class="detail" style="margin-top:12px;color:#bfcae1">0% · 0 MB · 0 MB/s · ETA —</div>
        <button type="button" class="retry" style="display:none;margin-top:12px;padding:8px 16px">Thử lại</button>`;
      const host = document.querySelector('#output-area') || document.body;
      host.appendChild(root);
      const fileInput=root.querySelector('input'), st=root.querySelector('.status'), bar=root.querySelector('.progress'), detail=root.querySelector('.detail'), retry=root.querySelector('.retry');
      const invoke = async (fn, args) => {
        const out = await google.colab.kernel.invokeFunction('realesrgan_' + id + '.' + fn, args, {});
        if (out?.data?.['application/json']?.error) throw Error(out.data['application/json'].error);
        return out;
      };
      const toBase64 = blob => new Promise((resolve,reject)=>{
        const reader = new FileReader();
        reader.onload=()=>resolve(reader.result);
        reader.onerror=()=>reject(Error('Không đọc được dữ liệu video'));
        reader.readAsDataURL(blob);
      });
      const fmt=x=>(x/(1024*1024)).toFixed(1);
      const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
      return await new Promise(resolve=>{
        let active=false;
        const start=async()=>{
          if(active) return;
          const f=fileInput.files?.[0]; if(!f)return;
          active=true;fileInput.disabled=true;retry.style.display='none';
          let sent=0, lastAt=performance.now(), lastBytes=0, rate=0;
          const started=performance.now();
          try{
            st.textContent='Đang mở phiên upload: '+f.name;
            await invoke('init',[f.name,f.size]);
            while(sent < f.size){
              const blob=f.slice(sent,Math.min(f.size,sent+chunkSize));
              const encoded=await toBase64(blob);
              // A completed kernel callback means that chunk is written to disk.
              await invoke('chunk',[sent,encoded]);
              sent+=blob.size;
              const now=performance.now();
              const dt=Math.max(0.05,(now-lastAt)/1000);
              const sample=(sent-lastBytes)/dt;
              rate=rate?rate*0.65+sample*0.35:sample;
              lastAt=now;lastBytes=sent;
              const pct=100*sent/f.size;
              bar.style.width=pct.toFixed(2)+'%';
              st.textContent='Đã xác nhận '+fmt(sent)+' / '+fmt(f.size)+' MB';
              detail.textContent=pct.toFixed(1)+'% · '+fmt(sent)+'/'+fmt(f.size)+' MB · '+fmt(rate)+' MB/s · ETA '+(rate>0?Math.ceil((f.size-sent)/rate)+' giây':'—');
              await wait(0);
            }
            await invoke('finish',[]);
            st.textContent='✓ Upload hoàn thành. Bắt đầu khôi phục video...';
            detail.textContent='100% · '+fmt(f.size)+' MB đã lưu trên Colab';
            resolve({ok:true});
          }catch(err){
            st.textContent='✗ Lỗi tải video: '+String(err?.message||err);
            detail.textContent='Đã xác nhận '+fmt(sent)+' MB. Chọn chạy lại ô để tạo phiên mới.';
            resolve({ok:false,error:String(err?.message||err)});
          }
        };
        fileInput.addEventListener('change',start);
      });
    })()'''.replace('__SESSION__', json.dumps(session)).replace('__CHUNK__', str(CHUNK_MAX))

    try:
        result = output.eval_js(code, ignore_result=False, timeout_sec=36000)
        if not result or not result.get('ok') or not state['done']:
            raise RuntimeError('Tải video chưa hoàn thành: ' + str((result or {}).get('error', 'không rõ')))
        return state['path']
    finally:
        # Kernel callbacks are session-specific; no public HTTP endpoint is started.
        pass
