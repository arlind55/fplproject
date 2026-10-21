// Shared header logic for every data page: the "Data updated …" status line.
// Each page loads this before its own script, so the gameweek state reads the same everywhere.

// fixtures: fixtures_lite.csv (or fixtures.csv) rows → {gw: {played, total}}
function gwProgress(fixtures){
  const truthy=v=>/^(true|1)$/i.test(String(v)),int=v=>v===""||v==null?null:+v;
  const prog={};
  for(const f of fixtures||[]){
    const gw=int(f.event);if(!gw)continue;
    const hs=int(f.team_h_score),as=int(f.team_a_score);
    const played=truthy(f.finished)||truthy(f.finished_provisional)||(hs!==null&&as!==null&&truthy(f.started));
    prog[gw]=prog[gw]||{played:0,total:0};prog[gw].total++;if(played)prog[gw].played++;
  }
  return prog;
}

// events: events.csv rows; meta: meta.json or null; fixtures: fixture rows, or null when the page has none
function describeData(events,meta,fixtures){
  const truthy=v=>/^(true|1)$/i.test(String(v));
  const fin=events.filter(e=>truthy(e.finished)).map(e=>+e.id);const lastDone=fin.length?Math.max(...fin):0;
  const cur=events.find(e=>truthy(e.is_current)),next=events.find(e=>truthy(e.is_next));
  let when="";if(meta?.updated_at){const d=new Date(meta.updated_at);when=d.toLocaleString([],{day:"2-digit",month:"short",hour:"2-digit",minute:"2-digit"});}
  let gw=`<b>GW${lastDone}</b> complete`;
  if(cur&&!truthy(cur.finished)&&+cur.id>lastDone){
    const p=fixtures?gwProgress(fixtures)[+cur.id]:null;
    gw=p&&p.played>=p.total?`<b>GW${cur.id}</b> played · awaiting FPL confirmation`:`<b>GW${cur.id}</b> in progress${p?` (${p.played}/${p.total} played)`:""}`;
  }
  return `Updated ${when?`<b>${when}</b>`:"<b>—</b>"} · ${gw} · GW${next?next.id:"?"} next`;
}
