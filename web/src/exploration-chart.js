// 为 x 轴标签和区间选择器各自预留空间，完整标签仍在悬浮提示中展示。
export function distributionOption(result){
  const rows=result?.distribution||[],date=result?.kind==='date'
  return {
    grid:{left:65,right:30,top:35,bottom:118},
    xAxis:{type:'category',data:rows.map(r=>r['区间']),boundaryGap:!date,axisLabel:{rotate:25,margin:12,width:100,overflow:'truncate',hideOverlap:true}},
    yAxis:{type:'value',name:'样本数',minInterval:1},
    dataZoom:[{type:'inside'},{type:'slider',left:65,right:30,bottom:12,height:20,showDetail:false}],
    series:[{name:'样本数',type:date?'line':'bar',data:rows.map(r=>r['样本数']),...(date?{showSymbol:rows.length<=30,areaStyle:{opacity:.1}}:{itemStyle:{borderRadius:0}})}],
  }
}

// 直接投影已返回的 summary 行；不创建第二套统计口径，多标签保留分组。
export function summaryRows(result,feature){
  const columns=result?.summary_columns||[]
  const flat=columns.flatMap(c=>c.children?c.children.map(child=>({...child,group:c.label})):[{...c,group:''}])
  const fieldKey=flat.find(c=>c.label==='特征名')?.key||'特征名'
  const row=(result?.summary||[]).find(row=>row[fieldKey]===feature)
  if(!row)return []
  const fields=flat.length?flat:Object.keys(row).map(key=>({key,label:key,group:''}))
  return fields.filter(c=>c.key!==fieldKey&&Object.hasOwn(row,c.key)).map(c=>({group:c.group,name:c.label,value:row[c.key]}))
}
