// 每个字段和字段组合独立选择呈现方式，图与表共享同一次运行的计数。
const formats={
  numeric:[['bar','直方图'],['box','箱线图'],['cumulative','累计占比曲线']],
  text:[['bar','长度直方图'],['box','长度箱线图'],['cumulative','长度累计占比']],
  date:[['line','时间趋势'],['bar','分期柱状图'],['cumulative','累计样本占比']],
  category:[['rank','频数条形图'],['pie','类别占比图']],
  boolean:[['pie','真假占比图'],['rank','频数条形图']],
}
export const kindNames={numeric:'数值',date:'日期',text:'自由文本',category:'类别',boolean:'布尔'}
export function columnKind(c){
  if(!c)return 'category'
  if(/bool/i.test(c.dtype))return 'boolean'
  if(c.kind==='category'&&c.unique>30&&!/category/i.test(c.dtype))return 'text'
  return c.kind
}
export const fieldChoices=kind=>[...(formats[kind]||formats.category),['table','分布表']]
export function pairChoices(kinds){
  const choices=[]
  const measures=kinds.filter(kind=>['numeric','text'].includes(kind)).length
  if(measures===1)choices.push(['means',kinds.includes('date')?'分期均值趋势':'分组均值图'])
  if(kinds.every(k=>['boolean','category'].includes(k)))choices.push(['stack','堆叠频数图'])
  if(measures===0)choices.push(['density','密度热力图'])
  choices.push(['scatter','散点图'])
  if(measures>0)choices.push(['density','密度热力图'])
  choices.push(['table','透视交叉表'])
  return choices
}
export const chosenFormat=(choices,choice)=>choices.some(([id])=>id===choice)?choice:choices[0][0]
const category=names=>({type:'category',data:names,axisLabel:{hideOverlap:true,rotate:28}})
const counts={type:'value',name:'样本数',minInterval:1}
export function distributionRows(dist){
  const total=dist.distribution.reduce((n,r)=>n+r['样本数'],0)
  return dist.distribution.map(r=>({...r,'占比（%）':total?r['样本数']/total*100:null}))
}
export function fieldChart(dist,format){
  const names=dist.distribution.map(r=>r['区间']),values=dist.distribution.map(r=>r['样本数'])
  if(format==='pie')return {legend:{type:'scroll',bottom:0},tooltip:{trigger:'item'},series:[{type:'pie',radius:['35%','63%'],label:{show:names.length<=8,formatter:'{b} {d}%'},data:names.map((name,i)=>({name,value:values[i]}))}]}
  if(format==='box')return {tooltip:{trigger:'item'},xAxis:category([dist.feature]),yAxis:{type:'value',scale:true,name:dist.kind==='text'?'字符数':''},series:[{name:'最小值 / Q1 / 中位数 / Q3 / 最大值',type:'boxplot',data:dist.five_number?.length?[dist.five_number]:[]}]}
  if(format==='rank')return {grid:{left:100},xAxis:counts,yAxis:{type:'category',inverse:true,data:names,axisLabel:{width:78,overflow:'truncate'}},series:[{type:'bar',name:'样本数',data:values}]}
  let data=values
  if(format==='cumulative'){
    const total=values.reduce((sum,n)=>sum+n,0)
    let sum=0
    data=values.map(n=>{sum+=n;return total?sum/total*100:null})
  }
  return {xAxis:category(names),dataZoom:[{type:'inside'},{type:'slider',bottom:2,height:15,showDetail:false}],grid:{bottom:72},yAxis:format==='cumulative'?{type:'value',name:'累计占比（%）',max:100}:counts,
    series:[{name:format==='cumulative'?'累计占比（%）':'样本数',type:format==='bar'?'bar':'line',showSymbol:names.length<30,data}]}
}
function axis(a){
  return {name:a.unit?a.name+'（'+a.unit+'）':a.name,type:a.categories?'category':a.kind==='date'?'time':'value',data:a.categories,scale:true,nameLocation:'middle',nameGap:32}
}
export function crossCounts(pair,group){
  const cells=new Map()
  for(const row of pair.density){
    if(group!==null&&row['标签']!==group)continue
    const key=JSON.stringify([row.x,row.y]),cell=cells.get(key)
    if(cell)cell.count+=row['样本数']
    else cells.set(key,{x:row.x,y:row.y,count:row['样本数']})
  }
  return [...cells.values()]
}
export function pairChart(pair,axes,format,group=null){
  const [a,b]=pair.indices.map(i=>axes[i])
  const legend={type:'scroll',top:0}
  if(format==='scatter'){
    const series=new Map()
    for(const point of pair.points){
      if(group!==null&&point.label!==group)continue
      if(!series.has(point.label))series.set(point.label,[])
      series.get(point.label).push(point.values)
    }
    return {tooltip:{trigger:'item'},legend,xAxis:axis(a),yAxis:axis(b),dataZoom:[{type:'inside'}],series:[...series].map(([name,data])=>({name,type:'scatter',symbolSize:5,data}))}
  }
  if(format==='means'){
    const rows=pair.means.filter(r=>group===null||r['标签']===group)
    const keys=[...new Set(rows.map(r=>r['维度']))],labels=[...new Set(rows.map(r=>r['标签']))]
    const lookup=new Map(rows.map(r=>[JSON.stringify([r['维度'],r['标签']]),r['均值']]))
    const time=axes[pair.dimension].kind==='date',measure=axes[pair.measure]
    const series=labels.map(name=>{
      const means=keys.map(k=>lookup.get(JSON.stringify([k,name]))??null)
      return {name,type:time?'line':'bar',data:time?keys.map((k,i)=>[k,means[i]]):means}
    })
    return {legend,xAxis:time?{type:'time'}:category(keys),yAxis:{type:'value',name:measure.name+(measure.unit?'平均'+measure.unit:'均值')},series}
  }
  const [xs,ys]=pair.order,cells=crossCounts(pair,group)
  if(format==='stack'){
    const lookup=new Map(cells.map(r=>[JSON.stringify([r.x,r.y]),r.count]))
    return {legend,xAxis:category(xs),yAxis:counts,series:ys.map(name=>({name,stack:'频数',type:'bar',data:xs.map(x=>lookup.get(JSON.stringify([x,name]))||0)}))}
  }
  const ix=new Map(xs.map((x,i)=>[x,i])),iy=new Map(ys.map((y,i)=>[y,i]))
  return {tooltip:{trigger:'item'},grid:{bottom:90},xAxis:category(xs),yAxis:category(ys),visualMap:{min:0,max:cells.reduce((m,r)=>Math.max(m,r.count),1),orient:'horizontal',bottom:0,left:'center'},series:[{type:'heatmap',name:'样本数',data:cells.map(r=>[ix.get(r.x),iy.get(r.y),r.count])}]}
}
