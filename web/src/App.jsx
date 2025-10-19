import React, { useEffect, useRef, useState } from "react";
import { predict } from "./api";

export default function App() {
  const canvasRef = useRef(null);
  const [digit, setDigit] = useState(null);
  const [probs, setProbs] = useState([]);
  const [busy, setBusy] = useState(false);
  const size = 280;

  useEffect(() => {
    const c = canvasRef.current;
    c.width = size; c.height = size;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "black"; ctx.fillRect(0,0,c.width,c.height);
    ctx.lineWidth = 18; ctx.lineCap = "round"; ctx.strokeStyle = "white";

    let drawing = false;
    const down = e => { drawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); };
    const move = e => { if(drawing){ ctx.lineTo(e.offsetX, e.offsetY); ctx.stroke(); } };
    const up = () => { drawing = false; };

    c.addEventListener("mousedown", down);
    c.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    // Touch support
    c.addEventListener("touchstart", e => {
      const r = c.getBoundingClientRect();
      const t = e.touches[0];
      drawing = true; ctx.beginPath(); ctx.moveTo(t.clientX-r.left, t.clientY-r.top);
    }, {passive:true});
    c.addEventListener("touchmove", e => {
      if(!drawing) return;
      const r = c.getBoundingClientRect();
      const t = e.touches[0];
      ctx.lineTo(t.clientX-r.left, t.clientY-r.top); ctx.stroke();
    }, {passive:true});
    window.addEventListener("touchend", ()=> drawing=false);

    return () => {
      c.removeEventListener("mousedown", down);
      c.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    }
  }, []);

  const clear = () => {
    const c = canvasRef.current, ctx = c.getContext("2d");
    ctx.fillStyle = "black"; ctx.fillRect(0,0,c.width,c.height);
    setDigit(null); setProbs([]);
  };

  const runPredict = async () => {
    try {
      setBusy(true);
      const dataUrl = canvasRef.current.toDataURL("image/png");
      const out = await predict(dataUrl);
      setDigit(out.digit); setProbs(out.probs || []);
    } catch (e) {
      alert(e.message || "Prediction failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{display:"grid",placeItems:"center",minHeight:"100dvh",gap:16,fontFamily:"system-ui"}}>
      <h1>MNIST Digit Classifier</h1>
      <canvas ref={canvasRef} style={{border:"1px solid #ccc", touchAction:"none"}} />
      <div style={{display:"flex",gap:8}}>
        <button onClick={clear}>Clear</button>
        <button onClick={runPredict} disabled={busy}>{busy ? "Predicting..." : "Predict"}</button>
      </div>
      {digit !== null && (
        <div style={{textAlign:"center"}}>
          <h2>Prediction: {digit}</h2>
          <div style={{maxWidth:440}}>
            <ol style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr 1fr",gap:6,listStyle:"none",padding:0}}>
              {probs.map((p,i)=>(
                <li key={i} style={{textAlign:"center"}}>
                  <div style={{height:80, display:"flex", alignItems:"end"}}>
                    <div style={{width:"100%", height: `${Math.round(p*100)}%`, background:"#ddd"}}/>
                  </div>
                  <div style={{marginTop:4,fontSize:12}}>{i} ({(p*100).toFixed(0)}%)</div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
      <p style={{fontSize:12,opacity:.7}}>Tip: draw big, centered strokes.</p>
    </div>
  );
}