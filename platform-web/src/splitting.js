export const splitMethods={random:'随机划分',time:'时间划分',group:'分组划分',oot:'时间外划分（OOT）'}

export function initialSplit(dataset){
  const prefix=(dataset?.name||'数据集').slice(0,130)
  return {method:'random',random_state:42,stratify:null,time_col:dataset?.default_date||dataset?.columns.find(c=>c.kind==='date')?.name||null,group_col:null,time_mode:'ratio',cutoffs:[],gap:0,oot_start:null,oot_end:null,parts:[{name:`${prefix}_训练集`,percent:60},{name:`${prefix}_验证集`,percent:20},{name:`${prefix}_测试集`,percent:20}]}
}

export function splitPayload(form,revision){
  const ratio=form.method!=='oot'&&!(form.method==='time'&&form.time_mode==='cutoff')
  const result={method:form.method,expected_revision:revision,parts:form.parts.map(p=>({name:p.name.trim(),...(ratio?{ratio:p.percent/100}:{})}))}
  if(['random','group'].includes(form.method))result.random_state=form.random_state
  if(form.method==='random')result.stratify=form.stratify||null
  if(form.method==='group')result.group_col=form.group_col
  if(['time','oot'].includes(form.method))Object.assign(result,{time_col:form.time_col,gap:form.gap})
  if(form.method==='time')Object.assign(result,{time_mode:form.time_mode,cutoffs:form.time_mode==='cutoff'?form.cutoffs.slice(0,form.parts.length-1):[]})
  if(form.method==='oot')Object.assign(result,{oot_start:form.oot_start,oot_end:form.oot_end||null})
  return result
}

export function splitError(form,existing=[]){
  const names=form.parts.map(p=>p.name.trim())
  if(names.length<2||names.length>10)return '请配置 2 到 10 个结果数据集'
  if(names.some(n=>!n||n.length>150))return '每个结果名称需要 1 到 150 个字符'
  if(new Set(names).size!==names.length)return '结果数据集名称不能重复'
  if(names.some(n=>existing.includes(n)))return '结果名称与已有数据集重复，请修改名称'
  if(form.method==='oot'&&names.length!==2)return '时间外划分需要两个结果数据集'
  const ratio=form.method!=='oot'&&!(form.method==='time'&&form.time_mode==='cutoff')
  if(ratio&&(form.parts.some(p=>!Number.isFinite(p.percent)||p.percent<=0||p.percent>=100)||Math.abs(form.parts.reduce((sum,p)=>sum+p.percent,0)-100)>0.000001))return '每份比例必须大于 0，且合计为 100%'
  if(['random','group'].includes(form.method)&&(!Number.isInteger(form.random_state)||form.random_state<0||form.random_state>2147483647))return '随机种子需要为 0 到 2147483647 的整数'
  if(form.method==='group'&&!form.group_col)return '请选择分组字段'
  if(['time','oot'].includes(form.method)){
    if(!form.time_col)return '请选择时间字段'
    if(!Number.isInteger(form.gap)||form.gap<0)return '边界间隔需要为非负整数行数'
  }
  if(form.method==='oot'){
    if(!form.oot_start)return '请选择时间外起始时间'
    if(form.oot_end&&form.oot_end<form.oot_start)return '时间外结束时间不能早于起始时间'
  }
  if(form.method==='time'&&form.time_mode==='cutoff'){
    const boundaries=form.cutoffs.slice(0,names.length-1)
    if(boundaries.length!==names.length-1||boundaries.some(v=>!v))return '请填写相邻数据集之间的时间边界'
    if(boundaries.some((v,i)=>i>0&&v<=boundaries[i-1]))return '时间边界需要从早到晚且不能重复'
  }
  return ''
}
