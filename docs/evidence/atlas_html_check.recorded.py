"""Exercise the actual embedded HTML and its private recipe handoff."""
import sys
sys.path.insert(0,"/home/cody/tkintegers")
import json
import time
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from tkintegers_ui import ConverterUI

w=tk.Tk()
app=ConverterUI(w)
app.tabs.select(3)
received=[]
original=app.atlas_panel.handle
def handle(event):
    received.append(event)
    print(json.dumps({k:v for k,v in event.items() if k!='plan'}),flush=True)
    original(event)
app.atlas_panel.handle=handle
stage=0
deadline=time.monotonic()+40
path=Path('/tmp/opencode/atlas-html-handoff.json')
dialog=patch('tkintegers_ui.filedialog.asksaveasfilename',return_value=str(path))
dialog.start()
def check():
    global stage
    try:
        if time.monotonic()>deadline:
            raise AssertionError('Timed out waiting for real HTML UI')
        loaded=next((e for e in received if e.get('type')=='loaded'),None)
        if stage==0 and loaded:
            assert loaded['ok'],loaded
            assert loaded['dom']['nodes']>10 and loaded['dom']['models']==10,loaded
            app.atlas_panel.send({'type':'evaluate','id':'click', 'code':"document.querySelector('#flowSvg [data-node]').dispatchEvent(new MouseEvent('click',{bubbles:true})); JSON.stringify({dialog:document.querySelector('#layerChoice').open, text:document.querySelector('#layerChoiceDescription').textContent, adjacent:document.querySelector('#importPlanButton').nextElementSibling.id})"})
            stage=1
        elif stage==1:
            result=next((e for e in received if e.get('id')=='click'),None)
            if result:
                data=json.loads(result['result'])
                assert data['dialog'] and len(data['text'])>20,data
                assert data['adjacent']=='reviewInTkIntegers',data
                app.atlas_panel.send({'type':'evaluate','id':'review','code':"document.querySelector('#layerChoice').close(); document.querySelector('#flowSvg [data-node=qkv]').dispatchEvent(new MouseEvent('click',{bubbles:true})); document.querySelector('#graphPrecision').value='TKinteger-k3'; document.querySelector('#graphPrecision').dispatchEvent(new Event('change',{bubbles:true})); document.querySelector('#layerChoice').close(); document.querySelector('#reviewInTkIntegers').click(); 'clicked'"})
                stage=2
        elif stage==2 and app.atlas_recipe:
            assert path.exists()
            assert app.tabs.index(app.tabs.select())==0
            assert app.recipe_table.get_children()
            assert app.convert_button.instate(['disabled'])
            assert any(r['precision']=='TKinteger-k3' for r in app.atlas_recipe['precision'])
            app.tabs.select(3)
            plan=next(e['plan'] for e in received if e.get('type')=='review')
            source=json.dumps(json.dumps(plan))
            app.atlas_panel.send({'type':'evaluate','id':'import','code':f"window.atlasImportDone=false; const input=document.querySelector('#planFile'); const dt=new DataTransfer(); dt.items.add(new File([{source}], 'roundtrip.json',{{type:'application/json'}})); input.files=dt.files; input.onchange({{target:input}}).then(()=>{{window.atlasImportDone=true;}}); 'importing'"})
            stage=3
        elif stage==3:
            app.atlas_panel.send({'type':'evaluate','id':'import-result','code':"JSON.stringify({done:!!window.atlasImportDone, notice:document.querySelector('#notice').textContent})"})
            stage=4
        elif stage==4:
            results=[e for e in received if e.get('id')=='import-result']
            if not results:
                w.after(100,check)
                return
            data=json.loads(results[-1]['result'])
            if not data['done']:
                stage=3
                w.after(100,check)
                return
            assert data['notice'].startswith('Imported roundtrip.json'),data
            print('PASS: original HTML, ten presets, SVG flow, real layer edit/dialog, adjacent handoff, saved-file review, original HTML recipe import, no conversion',flush=True)
            app.atlas_panel.send({'type':'evaluate','id':'diagram','code':"document.querySelector('#viewport').scrollIntoView({block:'center'}); 'diagram'"})
            dialog.stop()
            w.after(90000,app.close)
            return
        w.after(100,check)
    except Exception:
        dialog.stop()
        app.close()
        raise
w.after(100,check)
w.mainloop()
