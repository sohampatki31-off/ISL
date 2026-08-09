// ── Reference data ────────────────────────────────────────────
const ALPHA = [
  {l:'A',e:'✊✊',d:'Tips of both index fingers touching'},
  {l:'B',e:'🖐🖐',d:'Both hands curved, double circle'},
  {l:'C',e:'🤏',d:'Right hand curved into C'},
  {l:'D',e:'☝️🤏',d:'Left index up; right curves to touch'},
  {l:'E',e:'☝️👉',d:'Left index up; right touches tip'},
  {l:'F',e:'✌️✌️',d:'Left index/middle; right crosses over'},
  {l:'G',e:'✊✊',d:'Both fists, one on top of other'},
  {l:'H',e:'🖐🖐',d:'Left flat; right palm rubs across'},
  {l:'I',e:'☝️',d:'Right index points to cheek'},
  {l:'J',e:'☝️🖐',d:'Right index traces J on left palm'},
  {l:'K',e:'☝️👉',d:'Left index up; right bent touches it'},
  {l:'L',e:'🤙',d:'Left index and thumb form L'},
  {l:'M',e:'🖐🖐',d:'Left palm flat; three fingertips touch'},
  {l:'N',e:'🖐🖐',d:'Left palm flat; two fingertips touch'},
  {l:'O',e:'👌',d:'Index and thumb form circle O'},
  {l:'P',e:'☝️👌',d:'Left index up; right circle touches tip'},
  {l:'Q',e:'👌👉',d:'Left circle; right hooks inside'},
  {l:'R',e:'🖐👉',d:'Left palm flat; right index crosses'},
  {l:'S',e:'☝️✊',d:'Left index up; right wraps around wrist'},
  {l:'T',e:'☝️👉',d:'Left index up; right touches side'},
  {l:'U',e:'✌️👉',d:'Left index/middle up; right touches tips'},
  {l:'V',e:'✌️👉',d:'Left V; right index touches cleft'},
  {l:'W',e:'🖐👉',d:'Three fingers up; right touches tips'},
  {l:'X',e:'👉👉',d:'Both index fingers crossed'},
  {l:'Y',e:'☝️👉',d:'Left index up; right points to cleft'},
  {l:'Z',e:'🖐👉',d:'Left index up; right points to palm'},
];
const GESTS = [
  {l:'HELLO',    e:'👋',d:'Open palm wave'},
  {l:'LOVE',     e:'🤟',d:'ILY hand shape'},
  {l:'THANKYOU', e:'🙏',d:'Flat hand moves from chin'},
  {l:'HELP',     e:'🆘',d:'Fist lifts on palm'},
  {l:'SORRY',    e:'✊',d:'Fist circles heart'},
  {l:'YES',      e:'✅',d:'Fist nods up and down'},
  {l:'NO',       e:'❌',d:'Index/middle open and close'},
  {l:'GOOD',     e:'👍',d:'Thumbs up gesture'},
  {l:'PLEASE',   e:'🙏',d:'Flat hand circles chest'},
];
const ALL_CLASSES = [...ALPHA.map(a=>a.l), ...GESTS.map(g=>g.l)];
const WORD_GESTS = new Set(GESTS.map(g=>g.l));

// ── Build reference panel ─────────────────────────────────────
(function(){
  const rb = document.getElementById('ref-body');
  let h = '<div class="sec-lbl">A – Z FINGERSPELLING</div><div class="ref-grid">';
  ALPHA.forEach(a=>{
    h+=`<div class="ref-card" id="card-${a.l}" title="${a.d}"><div class="char">${a.l}</div><div class="emoji">${a.e}</div><div class="desc">${a.d}</div></div>`;
  });
  h+='</div><div class="sec-lbl" style="margin-top:12px">SPECIAL GESTURES</div><div class="gest-grid">';
  GESTS.forEach(g=>{
    h+=`<div class="gest-card" id="card-${g.l}" title="${g.d}"><div class="emoji">${g.e}</div><div class="label">${g.l}</div><div class="desc">${g.d}</div></div>`;
  });
  h+='</div>';
  rb.innerHTML=h;
})();

// ── Build trainer class dropdown ──────────────────────────────
(function(){
  const sel = document.getElementById('class-sel');
  ALL_CLASSES.forEach(c=>{
    const o=document.createElement('option');
    o.value=c; o.textContent=c; sel.appendChild(o);
  });
})();

// ── Tab switching ─────────────────────────────────────────────
let activeTab = 'recognize';
function switchTab(id){
  activeTab = id;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.querySelector(`.tab-btn[onclick="switchTab('${id}')"]`).classList.add('active');
  document.getElementById(id+'-tab').classList.add('active');
  setActiveFeed(id);
  if(id==='train') fetchStats();
}

// ── Loading screen helpers (registered BEFORE socket init so they always fire) ──
function _hideLS(){const e=document.getElementById('ls');if(e&&!e.classList.contains('hidden'))e.classList.add('hidden');}
setTimeout(_hideLS,5000);

// ── Video feed via MJPEG stream, state via plain HTTP polling ──
// /video_feed is a single persistent multipart/x-mixed-replace connection:
// the browser repaints the <img> itself as each part arrives, so there's
// no per-frame HTTP round-trip like the old /frame.jpg polling had. State
// (letters, sentence, etc.) still comes from /state — no Socket.IO, which
// sidesteps the Socket.IO/Engine.IO version mismatch that was blocking
// the connection on this machine.
const cvRec = document.getElementById('cv-rec');
const cvTrn = document.getElementById('cv-train');

function _openFeed(img){
  img.onload  = _hideLS;
  img.onerror = ()=>{
    setTimeout(()=>{ if(img.src.indexOf('/video_feed')!==-1) img.src = '/video_feed?_=' + Date.now(); }, 800);
  };
  img.src = '/video_feed?_=' + Date.now();
}
function _closeFeed(img){
  img.onerror = null;
  img.removeAttribute('src');
}

function setActiveFeed(id){
  const next = (id==='train') ? cvTrn : cvRec;
  const prev = (next===cvTrn) ? cvRec : cvTrn;
  if(prev.src) _closeFeed(prev);
  _openFeed(next);
}

setActiveFeed(activeTab || 'recognize');

// ── Apply a /state snapshot to the UI ─────────────────────────
let prevLetter='', flashTmr, prevCard=null, _lastTtsId=0, _ttsInit=false;

let _prevState = {};   // cache for change-detection to avoid per-poll DOM thrash
let currentFullText = '';

function applyState(d){
  // Speak any new utterance (works for manual actions and auto-speak).
  if(d.tts_id !== undefined){
    if(!_ttsInit){ _lastTtsId = d.tts_id; _ttsInit = true; }  // don't replay backlog on load
    else if(d.tts_id > _lastTtsId){ _lastTtsId = d.tts_id; if(d.tts_text) speakText(d.tts_text); }
  }

  // Model badge (only on change)
  if(d.model_features_loaded!==_prevState.model_features_loaded){
    const is2h=d.model_features_loaded===132;
    document.getElementById('class-count').textContent=is2h?'2-Handed':'1-Handed';
    document.getElementById('model-feat').innerHTML=`<b>${d.model_features_loaded}</b> feat`;
    const wb=document.getElementById('warn-r');
    if(!is2h){wb.style.display='flex';document.getElementById('warn-txt-r').textContent='⚠ Old model — retrain with 2-handed data';}
    else wb.style.display='none';
  }

  // Hand indicators (only on change)
  if(d.left_hand_detected!==_prevState.left_hand_detected || d.right_hand_detected!==_prevState.right_hand_detected){
    [['lh-r','lh-t','left_hand_detected'],['rh-r','rh-t','right_hand_detected']].forEach(([a,b,k])=>{
      [a,b].forEach(id=>{const el=document.getElementById(id);if(el)el.classList.toggle('on',d[k]);});
    });
  }

  // Letter
  const le=document.getElementById('letter');
  if(d.letter!==prevLetter){le.textContent=d.letter||'—';le.className='letter-char'+(d.letter?'':' empty');prevLetter=d.letter;}
  if(d.letter_flash){le.classList.remove('flash');void le.offsetWidth;le.classList.add('flash');clearTimeout(flashTmr);flashTmr=setTimeout(()=>le.classList.remove('flash'),460);}

  // Bars (only on change)
  const cf=Math.round(d.confidence*100);
  if(cf!==_prevState._cf){
    const cb=document.getElementById('conf-bar');
    cb.style.width=cf+'%'; cb.className='mfill '+(cf>70?'em':cf>40?'amber':'rose');
    document.getElementById('conf-val').textContent=cf+'%';
    _prevState._cf=cf;
  }
  if(d.stable_count!==_prevState.stable_count){
    document.getElementById('stab-bar').style.width=(d.stable_count/8*100)+'%';
    document.getElementById('stab-val').textContent=d.stable_count+'/8';
  }

  // Sparkline
  drawSpark(d.confidence||0);

  // Current word (only on change)
  if(d.current_word!==_prevState.current_word){
    const wd=document.getElementById('word-disp');
    wd.textContent=d.current_word||'_'; wd.className='word-disp'+(d.current_word?'':' empty');
  }

  // Suggestions (only on change)
  const sugKey=(d.suggestions||[]).join('|');
  if(sugKey!==_prevState._sugKey){
    document.getElementById('sugs').innerHTML=(d.suggestions||[]).map((s,i)=>
      `<button class="sug-pill" onclick="act('suggest',${i})"><span class="hint">${i+1}</span>${s}</button>`
    ).join('');
    _prevState._sugKey=sugKey;
  }

  // ── Live sentence (full text + the word being signed) ──
  if(d.full_text!==_prevState.full_text || d.current_word!==_prevState.current_word){
    currentFullText=((d.full_text||'')+(d.current_word||'')).trim();
    const box=document.getElementById('full-text-box');
    if(!d.full_text && !d.current_word){
      box.className='text-box empty';
      box.textContent='Sign letters to build your sentence — it appears here.';
    } else {
      box.className='text-box';
      const safe=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      box.innerHTML=safe(d.full_text||'')+(d.current_word?`<span class="pending">${safe(d.current_word)}</span>`:'');
      box.scrollTop=box.scrollHeight;   // keep latest text in view
    }
    const st=document.getElementById('sent-txt');
    if(st){ st.textContent=currentFullText||'Sign letters to build a sentence...'; st.className='sent-box'+(currentFullText?' has':' empty'); }
  }

  // Active ref card (only on letter change)
  if(d.letter!==_prevState.letter){
    if(prevCard){const old=document.getElementById('card-'+prevCard);if(old)old.classList.remove('active');}
    if(d.letter){const card=document.getElementById('card-'+d.letter);if(card){card.classList.add('active');prevCard=d.letter;const rb=document.getElementById('ref-body');const cr=card.getBoundingClientRect(),rr=rb.getBoundingClientRect();if(cr.top<rr.top||cr.bottom>rr.bottom)card.scrollIntoView({behavior:'smooth',block:'nearest'});}}
  }

  // Session stats (only on change)
  if(d.total_letters!==undefined && d.total_letters!==_prevState.total_letters){const el=document.getElementById('stat-ltrs');if(el)el.textContent=d.total_letters;}

  // History (only on change)
  if(d.history){
    const hk=d.history.join('|');
    if(hk!==_prevState._hk){ renderHist(d.history); _prevState._hk=hk; }
  }

  // Confetti on word gesture
  if(d.letter_flash&&WORD_GESTS.has(d.letter)) launchConfetti();

  // remember the keys we diff on
  _prevState.model_features_loaded=d.model_features_loaded;
  _prevState.left_hand_detected=d.left_hand_detected;
  _prevState.right_hand_detected=d.right_hand_detected;
  _prevState.stable_count=d.stable_count;
  _prevState.current_word=d.current_word;
  _prevState.full_text=d.full_text;
  _prevState.letter=d.letter;
  _prevState.total_letters=d.total_letters;
}

async function pollState(){
  try{
    const r=await fetch('/state',{cache:'no-store'});
    applyState(await r.json());
  }catch(e){ /* transient; keep polling */ }
  setTimeout(pollState, 120);   // ~8 polls/sec
}
pollState();

// ── Training status polling ───────────────────────────────────
function applyTraining(data){
  const se=document.getElementById('train-stat');
  const ai=document.getElementById('acc-row');
  const tb=document.getElementById('train-btn');
  if(!se||!tb) return;
  if(data.status==='training'){
    se.innerHTML='<span style="color:var(--amber)">Training...</span>';
    tb.disabled=true; tb.innerHTML='<div class="spinner"></div> TRAINING';
  } else if(data.status==='success'){
    se.innerHTML='<span style="color:var(--emerald)">Success — model loaded</span>';
    const f1Row=document.getElementById('f1-row');
    const noteRow=document.getElementById('eval-note-row');
    if(data.macro_f1!=null){
      if(ai) ai.style.display='flex';
      document.getElementById('train-acc').textContent=(data.accuracy*100).toFixed(1)+'%';
      if(f1Row){ f1Row.style.display='flex'; document.getElementById('train-f1').textContent=data.macro_f1.toFixed(3); }
    } else {
      if(ai) ai.style.display='none';
      if(f1Row) f1Row.style.display='none';
    }
    if(noteRow && data.eval_note){ noteRow.style.display='flex'; document.getElementById('eval-note-txt').textContent=data.eval_note; }
    else if(noteRow){ noteRow.style.display='none'; }
    if(tb.disabled){ tb.disabled=false; tb.textContent='TRAIN MODEL'; fetchStats(); }
    else { tb.disabled=false; tb.textContent='TRAIN MODEL'; }
  } else if(data.status==='error'){
    se.innerHTML=`<span style="color:var(--rose)" title="${data.error}">Error: ${(data.error||'').substring(0,20)}…</span>`;
    tb.disabled=false; tb.textContent='TRAIN MODEL';
  } else { se.textContent='Idle'; tb.disabled=false; tb.textContent='TRAIN MODEL'; }
}

async function pollTraining(){
  try{
    const r=await fetch('/training_status',{cache:'no-store'});
    applyTraining(await r.json());
  }catch(e){}
  setTimeout(pollTraining, 900);
}
pollTraining();

// ── Session timer ─────────────────────────────────────────────
const _t0=Date.now();
setInterval(()=>{
  const s=Math.floor((Date.now()-_t0)/1000);
  const el=document.getElementById('sess-timer');
  if(el) el.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
},1000);

// ── Sparkline ─────────────────────────────────────────────────
const _spData=[];
const _spCv=document.getElementById('sparkline-canvas');
const _spCtx=_spCv?_spCv.getContext('2d'):null;
function drawSpark(v){
  if(!_spCtx)return;
  _spData.push(v); if(_spData.length>50)_spData.shift();
  const W=_spCv.offsetWidth||200,H=28;
  _spCv.width=W; _spCv.height=H;
  _spCtx.clearRect(0,0,W,H);
  const col=v>0.7?'#22C55E':v>0.4?'#E8A225':'#F43F5E';
  const fill=v>0.7?'rgba(34,197,94,.1)':v>0.4?'rgba(232,162,37,.1)':'rgba(244,63,94,.1)';
  const n=_spData.length;
  _spCtx.beginPath();
  _spData.forEach((p,i)=>{const x=i/(n-1||1)*W,y=H-p*(H-3);i===0?_spCtx.moveTo(x,y):_spCtx.lineTo(x,y);});
  _spCtx.lineTo(W,H); _spCtx.lineTo(0,H); _spCtx.closePath();
  _spCtx.fillStyle=fill; _spCtx.fill();
  _spCtx.beginPath();
  _spData.forEach((p,i)=>{const x=i/(n-1||1)*W,y=H-p*(H-3);i===0?_spCtx.moveTo(x,y):_spCtx.lineTo(x,y);});
  _spCtx.strokeStyle=col; _spCtx.lineWidth=1.5; _spCtx.stroke();
  _spCtx.setLineDash([2,3]); _spCtx.strokeStyle='rgba(255,255,255,.1)'; _spCtx.lineWidth=.7;
  _spCtx.beginPath(); _spCtx.moveTo(0,H*.5); _spCtx.lineTo(W,H*.5); _spCtx.stroke();
  _spCtx.setLineDash([]);
}

// ── Web Speech API ────────────────────────────────────────────
let ttsVoices=[], ttsRate=1.0;
function loadVoices(){
  ttsVoices=window.speechSynthesis.getVoices();
  const sel=document.getElementById('voice-sel');
  if(!sel||!ttsVoices.length)return;
  const en=ttsVoices.filter(v=>v.lang.startsWith('en'));
  const ot=ttsVoices.filter(v=>!v.lang.startsWith('en'));
  let h='<option value="-1">System Default</option>';
  if(en.length){h+='<optgroup label="English">';en.forEach(v=>{h+=`<option value="${ttsVoices.indexOf(v)}">${v.name}</option>`;});h+='</optgroup>';}
  if(ot.length){h+='<optgroup label="Other">';ot.forEach(v=>{h+=`<option value="${ttsVoices.indexOf(v)}">${v.name} (${v.lang})</option>`;});h+='</optgroup>';}
  sel.innerHTML=h;
}
if('speechSynthesis' in window){window.speechSynthesis.onvoiceschanged=loadVoices; loadVoices();}

function onRateChange(v){ttsRate=parseFloat(v); document.getElementById('rate-val').textContent=parseFloat(v).toFixed(1)+'×';}

function speakText(text){
  if(!text||!('speechSynthesis' in window))return;
  window.speechSynthesis.cancel();
  const utt=new SpeechSynthesisUtterance(text);
  const vi=parseInt((document.getElementById('voice-sel')||{}).value||-1);
  if(vi>=0&&ttsVoices[vi])utt.voice=ttsVoices[vi];
  utt.rate=ttsRate;
  utt.onstart=()=>showSpeaking(true);
  utt.onend=utt.onerror=()=>showSpeaking(false);
  window.speechSynthesis.speak(utt);
}
function showSpeaking(on){
  const w=document.getElementById('wave-ind'), b=document.getElementById('speak-btn');
  if(w)w.style.display=on?'flex':'none';
  if(b){b.innerHTML=on?'&#9658; SPEAKING...':'&#9658; SPEAK ALL'; b.classList.toggle('speaking',on);}
}
async function speakAndCommit(){await act('enter');}

// ── History ───────────────────────────────────────────────────
let histItems=[];
function addHistLocal(t){if(!t)return; if(!histItems.length||histItems[0]!==t){histItems=[t,...histItems].slice(0,10);renderHist(histItems);}}
function renderHist(items){
  histItems=items;
  const l=document.getElementById('hist-list');
  if(!items||!items.length){l.innerHTML='<div class="hist-empty">No sentences spoken yet.</div>';return;}
  l.innerHTML=items.map((t,i)=>`<div class="hist-item"><span class="hist-n">${i+1}</span><span class="hist-t">${t}</span><button class="hist-play" onclick="speakText(histItems[${i}])">&#9658;</button></div>`).join('');
}
function toggleHist(){
  const p=document.getElementById('hist-panel'),b=document.getElementById('hist-btn');
  p.classList.toggle('open'); b.classList.toggle('active');
}
async function clearHistory(){await act('clear_history'); histItems=[]; renderHist([]);}

// ── Copy / Export ─────────────────────────────────────────────
async function copyClip(){
  const el=document.getElementById('sent-txt');
  if(el.classList.contains('empty'))return;
  try{await navigator.clipboard.writeText(el.textContent.trim()); showToast('Copied to clipboard');}
  catch{showToast('Copy failed — try Ctrl+C');}
}
function exportSess(){const a=document.createElement('a');a.href='/download_history';a.download='isl_session.txt';a.click();showToast('Session exported');}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2200);}

// ── Auto-speak ────────────────────────────────────────────────
async function setAutoSpeak(v){await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'auto_speak',value:v})});}

// ── Generic action ────────────────────────────────────────────
async function act(action,index){
  const body={action}; if(index!==undefined)body.index=index;
  await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}

// ── Quick phrases / Punct ─────────────────────────────────────
async function quickPhrase(t){await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'quick_phrase',text:t})});}
async function addPunct(c){await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'punct',char:c})});}

// ── Shortcuts modal ───────────────────────────────────────────
function toggleSC(){document.getElementById('sc-modal').classList.toggle('open');}

// ── Keyboard shortcuts ────────────────────────────────────────
let _recHK=false;
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.key===' '){e.preventDefault();act('space');}
  else if(e.key==='Enter'){speakAndCommit();}
  else if(e.key==='Backspace'){act('backspace');}
  else if(e.key==='1'){act('suggest',0);}
  else if(e.key==='2'){act('suggest',1);}
  else if(e.key==='3'){act('suggest',2);}
  else if(e.key==='?'){toggleSC();}
  else if((e.key==='r'||e.key==='R')&&activeTab==='train'&&!e.repeat&&!_recHK){_recHK=true;_manualStart();}
});
document.addEventListener('keyup',e=>{if((e.key==='r'||e.key==='R')&&_recHK){_recHK=false;_manualStop();}});

// ── Countdown Auto-Record ─────────────────────────────────────
let _cdTimer=null, _arTimer=null, _recActive=false;

async function triggerAutoRecord(){
  if(_recActive){
    // Stop mid-recording
    clearInterval(_arTimer); _arTimer=null; _recActive=false;
    await _stopRec(); resetRecUI();
    return;
  }
  if(_cdTimer){
    // Cancel countdown
    clearInterval(_cdTimer); _cdTimer=null; resetRecUI();
    return;
  }
  // Start 3s countdown
  const ring=document.getElementById('cd-ring');
  const numEl=document.getElementById('cd-num');
  const btn=document.getElementById('rec-btn');
  ring.classList.add('show');
  btn.querySelector('.rbig').textContent='CANCEL';
  btn.querySelector('.rsub').textContent='CLICK TO CANCEL';
  let count=3;
  numEl.textContent=count;

  _cdTimer=setInterval(async()=>{
    count--;
    if(count>0){
      numEl.style.animation='none'; void numEl.offsetWidth; numEl.style.animation='cd-pop .85s ease-out';
      numEl.textContent=count;
    } else {
      clearInterval(_cdTimer); _cdTimer=null;
      ring.classList.remove('show');
      // Launch recording
      await _startRec();
      _recActive=true;
      btn.querySelector('.rbig').textContent='RECORDING';
      btn.querySelector('.rsub').textContent='CLICK TO STOP';
      btn.classList.add('recording');
      const prog=document.getElementById('rec-prog');
      const fill=document.getElementById('rec-prog-fill');
      const lbl=document.getElementById('rec-prog-lbl');
      prog.classList.add('show');
      let elapsed=0; const dur=5000;
      _arTimer=setInterval(async()=>{
        elapsed+=100;
        fill.style.width=Math.min(elapsed/dur*100,100)+'%';
        lbl.textContent=(elapsed/1000).toFixed(1)+'s / 5.0s';
        if(elapsed>=dur){
          clearInterval(_arTimer); _arTimer=null; _recActive=false;
          await _stopRec(); resetRecUI();
          showToast('Recording complete ✓');
        }
      },100);
    }
  },1000);
}

function resetRecUI(){
  const btn=document.getElementById('rec-btn');
  const ring=document.getElementById('cd-ring');
  const prog=document.getElementById('rec-prog');
  btn.classList.remove('recording');
  btn.querySelector('.rbig').textContent='RECORD';
  btn.querySelector('.rsub').textContent='CLICK TO START';
  ring.classList.remove('show');
  prog.classList.remove('show');
  document.getElementById('rec-prog-fill').style.width='0%';
  document.getElementById('rec-ind').style.display='none';
}

// ── Signer/session identity for the recording burst ───────────
// Persisted across reloads so a signer doesn't have to retype it for
// every recording action within one sitting. Sent with every
// /start_recording call so each row in isl_dataset.csv is attributable
// to who recorded it and in which session — required for any honest
// signer- or session-disjoint evaluation later.
const _signerInput  = document.getElementById('signer-id-input');
const _sessionInput = document.getElementById('session-id-input');
const _signerWarn   = document.getElementById('signer-warn');
_signerInput.value  = localStorage.getItem('isl_signer_id')  || '';
_sessionInput.value = localStorage.getItem('isl_session_id') || '';
_signerInput.addEventListener('change', ()=>localStorage.setItem('isl_signer_id', _signerInput.value.trim()));
_sessionInput.addEventListener('change', ()=>localStorage.setItem('isl_session_id', _sessionInput.value.trim()));
function _recordingIdentity(){
  return { signer_id: _signerInput.value.trim(), session_id: _sessionInput.value.trim() };
}

// Surface known signer ids so two different people are less likely to
// accidentally pick the same one (which would silently merge their data —
// see docs/data_collection_protocol.md). This cannot fully prevent a
// collision, only make existing ids visible.
let _knownSigners = [];
async function _refreshKnownSigners(){
  try{
    const r = await fetch('/known_signers', {cache:'no-store'});
    _knownSigners = await r.json();
    const list = document.getElementById('known-signers-list');
    list.innerHTML = _knownSigners.map(s=>`<option value="${s}"></option>`).join('');
  }catch(e){ /* non-critical */ }
}
_refreshKnownSigners();
_signerInput.addEventListener('input', ()=>{
  const v = _signerInput.value.trim();
  if(!v){ _signerWarn.style.display='none'; return; }
  _signerWarn.style.display='block';
  _signerWarn.textContent = _knownSigners.includes(v)
    ? `"${v}" already has recordings — make sure that's really the same person.`
    : `"${v}" is a new signer id.`;
});

let _recActive2=false;
async function _startRec(){
  if(_recActive2)return; _recActive2=true;
  const cur=document.getElementById('class-sel').value;
  document.getElementById('rec-ind').style.display='flex';
  await fetch('/start_recording',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur, ..._recordingIdentity()})});
}
async function _stopRec(){
  if(!_recActive2)return; _recActive2=false;
  await fetch('/stop_recording',{method:'POST'});
  fetchStats();
  _refreshKnownSigners();
}

// Manual hold-to-record fallback (R key)
async function _manualStart(){
  if(_recActive||_cdTimer)return;
  _recActive2=true;
  const cur=document.getElementById('class-sel').value;
  document.getElementById('rec-btn').classList.add('recording');
  document.getElementById('rec-ind').style.display='flex';
  await fetch('/start_recording',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur, ..._recordingIdentity()})});
}
async function _manualStop(){
  if(!_recActive2)return; _recActive2=false;
  document.getElementById('rec-btn').classList.remove('recording');
  document.getElementById('rec-ind').style.display='none';
  await fetch('/stop_recording',{method:'POST'});
  fetchStats();
  _refreshKnownSigners();
}

// ── Dataset stats ─────────────────────────────────────────────
async function fetchStats(){
  try{
    const res=await fetch('/dataset_info');
    const counts=await res.json();
    const cur=document.getElementById('class-sel').value;
    document.getElementById('target-cnt').textContent=counts[cur]||0;
    const grid=document.getElementById('stats-grid');
    grid.innerHTML=ALL_CLASSES.map(c=>{
      const n=counts[c]||0;
      return `<div class="sbadge ${n>0?'has':''}"><div class="sl">${c}</div><div class="sc">${n}</div></div>`;
    }).join('');
  }catch(e){console.error(e);}
}
function onClassChange(){fetchStats();}
async function deleteLatestSample(){
  const cur=document.getElementById('class-sel').value;
  if(!confirm(`Delete latest sample for '${cur}'?`))return;
  const r=await fetch('/delete_latest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur})});
  const d=await r.json(); document.getElementById('target-cnt').textContent=d.count; fetchStats();
}
async function clearClassSamples(){
  const cur=document.getElementById('class-sel').value;
  if(!confirm(`Clear ALL samples for '${cur}'?`))return;
  await fetch('/clear_class',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur})});
  document.getElementById('target-cnt').textContent=0; fetchStats();
}
async function trainModel(){await fetch('/train',{method:'POST'});}

// ── Resolution switcher ───────────────────────────────────────
async function setResolution(w,h){
  const r=await fetch('/set_resolution',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({width:w,height:h})});
  const d=await r.json(); if(d.status!=='ok'){showToast('Resolution change failed');return;}
  document.getElementById('res-hd').className='res-btn'+(w===1280?' active':'');
  document.getElementById('res-tr').className='res-btn'+(w===640?' active tr-mode':'');
  document.getElementById('res-note').innerHTML=w===640?'Current: <b>640×480</b> — Recommended for training accuracy':'Current: <b>1280×720</b> (HD Preview)';
  showToast(`Resolution → ${w}×${h}`);
}

// ── Confetti ──────────────────────────────────────────────────
const _ccv=document.getElementById('conf-cv');
if(_ccv){_ccv.width=innerWidth;_ccv.height=innerHeight;addEventListener('resize',()=>{_ccv.width=innerWidth;_ccv.height=innerHeight;});}
let _cpieces=[];
function launchConfetti(){
  if(!_ccv)return;
  const ctx=_ccv.getContext('2d');
  const cols=['#E8A225','#00C4A7','#7C6FF7','#F43F5E','#22C55E','#F5C05A'];
  for(let i=0;i<90;i++){_cpieces.push({x:Math.random()*_ccv.width,y:-10,vx:(Math.random()-.5)*5,vy:Math.random()*3+2,w:Math.random()*9+4,h:Math.random()*6+3,color:cols[Math.floor(Math.random()*cols.length)],a:Math.random()*Math.PI*2,va:(Math.random()-.5)*.2,alpha:1});}
  (function step(){
    ctx.clearRect(0,0,_ccv.width,_ccv.height);
    _cpieces=_cpieces.filter(p=>p.alpha>.04);
    _cpieces.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;p.vy+=.09;p.a+=p.va;
      if(p.y>_ccv.height*.75)p.alpha-=.025;
      ctx.save();ctx.translate(p.x,p.y);ctx.rotate(p.a);ctx.globalAlpha=p.alpha;ctx.fillStyle=p.color;ctx.fillRect(-p.w/2,-p.h/2,p.w,p.h);ctx.restore();
    });
    if(_cpieces.length>0)requestAnimationFrame(step);
    else ctx.clearRect(0,0,_ccv.width,_ccv.height);
  })();
}

// Initial load
fetchStats();
