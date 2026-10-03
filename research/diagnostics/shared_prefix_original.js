// Original exact-arithmetic computation; algorithm body preserved.
// Requires the small store/text host shims in reproduce_shared_prefix.cjs.
// Exhaustive arithmetic toy: not an LLM or GPU experiment.
const requests=[{id:"b",prefix:"B",deadline:4},{id:"a1",prefix:"A",deadline:6},{id:"a2",prefix:"A",deadline:6}];
const pre={A:{load:4,recompute:10},B:{load:2,recompute:3}};
function perms(xs){if(xs.length<2)return [xs];return xs.flatMap((x,i)=>perms(xs.filter((_,j)=>i!==j)).map(p=>[x,...p]));}
const solutions=[];
for(const a of ["load","recompute"])for(const b of ["load","recompute"]){
 const modes={A:a,B:b},tasks={};
 for(const p of ["A","B"])tasks[p]={dur:pre[p][modes[p]],res:modes[p]==="load"?"P":"G",parents:[]};
 for(const r of requests)tasks[r.id]={dur:1,res:"G",parents:[r.prefix]};
 const ps=Object.keys(tasks).filter(k=>tasks[k].res==="P"),gs=Object.keys(tasks).filter(k=>tasks[k].res==="G");
 for(const po of perms(ps))for(const go of perms(gs)){
  const parents=Object.fromEntries(Object.entries(tasks).map(([k,v])=>[k,[...v.parents]]));
  for(const order of [po,go])for(let i=1;i<order.length;i++)parents[order[i]].push(order[i-1]);
  const start={},end={},remain=new Set(Object.keys(tasks));
  while(remain.size){let changed=false;for(const k of [...remain])if(parents[k].every(x=>x in end)){start[k]=Math.max(0,...parents[k].map(x=>end[x]));end[k]=start[k]+tasks[k].dur;remain.delete(k);changed=true;}if(!changed)break;}
  if(remain.size)continue;
  const ontime=requests.filter(r=>end[r.id]<=r.deadline).length;
  solutions.push({modes,po,go,start,end,ontime,makespan:Math.max(...Object.values(end))});
 }
}
solutions.sort((a,b)=>b.ontime-a.ontime||a.makespan-b.makespan);
store("toy_exhaustive",solutions);
text({validSchedules:solutions.length,optimum:solutions[0],allLoadBest:solutions.filter(x=>x.modes.A==="load"&&x.modes.B==="load")[0],numberOptimal:solutions.filter(x=>x.ontime===3).length});
