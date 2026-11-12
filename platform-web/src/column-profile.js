// 表头悬浮详情的图表与数值展示，计算结果保持服务端 summary 原值。
export const profileTypes={numeric:'数值',category:'文本',date:'时间',boolean:'布尔'}
export function profileChart(profile){
  const points=profile.distribution||[],labels=points.map(p=>p.label),counts=points.map(p=>p.count)
  const base={animation:false,grid:{left:50,right:20,top:25,bottom:60},tooltip:{trigger:'axis',confine:true,renderMode:'richText'}}
  if(profile.kind==='boolean')return {...base,tooltip:{trigger:'item',confine:true,renderMode:'richText',formatter:'{b}：{c}（{d}%）'},legend:{bottom:0},series:[{name:'样本占比',type:'pie',radius:['42%','68%'],center:['50%','43%'],label:{formatter:'{b}\n{d}%'},data:points.map(p=>({name:p.label,value:p.count}))}]}
  if(profile.kind==='category')return {...base,grid:{left:115,right:35,top:10,bottom:30},xAxis:{type:'value',minInterval:1,name:'样本数'},yAxis:{type:'category',inverse:true,data:labels,axisLabel:{width:100,overflow:'truncate'}},series:[{name:'样本数',type:'bar',data:counts,barMaxWidth:18}]}
  return {...base,xAxis:{type:'category',data:labels,boundaryGap:profile.kind!=='date',axisLabel:{rotate:25,hideOverlap:true,...(profile.kind==='numeric'?{formatter:(value,index)=>Number(points[index].lower?.toPrecision(3)??index).toString()}: {})}},yAxis:{type:'value',minInterval:1,name:'样本数'},series:[{name:'样本数',type:profile.kind==='date'?'line':'bar',data:counts,...(profile.kind==='date'?{showSymbol:points.length<30,areaStyle:{opacity:0.12}}:{barCategoryGap:'0%',barGap:'0%'})}]}
}
export function profileValue(name,value){
  if(value===null||value===undefined)return '—'
  if(name==='字段类型')return {numerical:'数值',categorical:'类别',datetime:'时间',constant:'常量',id:'标识'}[value]||value
  if(typeof value==='boolean')return value?'真':'假'
  if(typeof value==='number')return /率$|占比$/.test(name)?`${(value*100).toFixed(2)}%`:Number.isInteger(value)?value.toLocaleString():Number(value.toPrecision(6)).toString()
  if(typeof value==='object')return JSON.stringify(value)
  return String(value)
}
