"""Interactive 3D viewer (3Dmol.js) rendered inside Streamlit."""
import json

_VIEWER = """
<div style="font-family:Inter,system-ui,sans-serif">
  <div id="bar" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px">
    <select id="style">__OPTIONS__</select>
    <select id="color">
      <option value="element">Colour: element</option>
      <option value="polar">Colour: polar (N, O) vs nonpolar</option>
      <option value="chain">Colour: chain</option>
      <option value="spectrum">Colour: N→C spectrum</option>
    </select>
    <label><input type="checkbox" id="surf"> Surface</label>
    <label><input type="checkbox" id="spin"> Spin</label>
    <button id="reset">Reset view</button>
    <button id="png">Save PNG</button>
  </div>
  <div id="v" style="width:100%;height:__H__px;position:relative;border-radius:12px;
       border:1px solid #d9e2ec;background:__BG__"></div>
  <div style="font-size:12px;color:#627d98;margin-top:4px">
    Drag to rotate · scroll to zoom · right-drag to move</div>
</div>
<script>
function boot(){
  const data = __DATA__, fmt = "__FMT__", name = __NAME__;
  const el = document.getElementById("v");
  const viewer = $3Dmol.createViewer(el, {backgroundColor: "__BG__"});
  viewer.addModel(data, fmt);
  let surf = null;
  const polar = function(atom){ return (atom.elem==="N"||atom.elem==="O") ? "#0072B2" :
                                       (atom.elem==="H" ? "#d0d7de" : "#E69F00"); };
  function colorSpec(){
    const c = document.getElementById("color").value;
    if (c==="polar") return {colorfunc: polar};
    if (c==="chain") return {colorscheme: "chain"};
    if (c==="spectrum") return {color: "spectrum"};
    return {colorscheme: "Jmol"};
  }
  function apply(){
    const s = document.getElementById("style").value, c = colorSpec();
    viewer.setStyle({}, {});
    if (s==="cartoon") { viewer.setStyle({}, {cartoon: Object.assign({}, c)});
                         viewer.setStyle({hetflag: true}, {stick: Object.assign({radius:0.2}, c)}); }
    else if (s==="sphere") viewer.setStyle({}, {sphere: Object.assign({scale:0.9}, c)});
    else if (s==="line") viewer.setStyle({}, {line: c});
    else if (s==="ballstick") viewer.setStyle({}, {stick: Object.assign({radius:0.14}, c),
                                                   sphere: Object.assign({scale:0.25}, c)});
    else viewer.setStyle({}, {stick: Object.assign({radius:0.18}, c)});
    if (surf !== null) { viewer.removeSurface(surf); surf = null; }
    if (document.getElementById("surf").checked)
      surf = viewer.addSurface($3Dmol.SurfaceType.SAS, Object.assign({opacity:0.55}, c));
    viewer.render();
  }
  ["style","color","surf"].forEach(id => document.getElementById(id).onchange = apply);
  document.getElementById("spin").onchange = e => viewer.spin(e.target.checked ? "y" : false);
  document.getElementById("reset").onclick = () => { viewer.zoomTo(); viewer.render(); };
  document.getElementById("png").onclick = () => {
    const a = document.createElement("a"); a.href = viewer.pngURI();
    a.download = name.replace(/[^A-Za-z0-9_-]+/g, "_") + "_3D.png"; a.click(); };
  apply(); viewer.zoomTo(); viewer.render();
}
function load(srcs){
  if (!srcs.length) { document.getElementById("v").innerHTML =
      "<p style='padding:16px'>3D viewer could not load (offline?).</p>"; return; }
  const s = document.createElement("script"); s.src = srcs[0];
  s.onload = boot; s.onerror = () => load(srcs.slice(1));
  document.head.appendChild(s);
}
load(["https://cdn.jsdelivr.net/npm/3dmol@2/build/3Dmol-min.js",
      "https://3Dmol.org/build/3Dmol-min.js"]);
</script>
"""


def viewer_html(text: str, fmt: str, name: str, protein: bool, height=430,
                background="#ffffff") -> str:
    opts = [("cartoon", "Cartoon"), ("stick", "Sticks"), ("ballstick", "Ball & stick"),
            ("sphere", "Spheres"), ("line", "Lines")]
    if not protein:
        opts = [o for o in opts if o[0] != "cartoon"]
        opts.insert(0, opts.pop(1))          # ball & stick first for compounds
    options = "".join(f'<option value="{v}">Style: {t}</option>' for v, t in opts)
    return (_VIEWER.replace("__OPTIONS__", options)
            .replace("__DATA__", json.dumps(text))
            .replace("__FMT__", fmt)
            .replace("__NAME__", json.dumps(name))
            .replace("__H__", str(height))
            .replace("__BG__", background))
