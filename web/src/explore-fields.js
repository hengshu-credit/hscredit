// 探索字段的放置、交换和移除；坐标轴顺序即后台 features 的顺序。
export function placeField(mode, axes, group, role, field, columns){
  const next=[...axes]
  const origin=next.indexOf(field)
  let label=group
  const byName=new Map(columns.map(c=>[c.name,c]))
  if(!byName.has(field))return {error:'当前数据集中没有该字段'}
  if(role==='group'){
    if(origin>=0){
      if(label)next[origin]=label
      else next.splice(origin,1)
    }
    label=field
  }else{
    if(role>next.length)return {error:'请先放入前一坐标轴的字段'}
    if(origin>=0){
      if(role===next.length){
        next.splice(origin,1)
        next.push(field)
      }else [next[role],next[origin]]=[next[origin],next[role]]
    }else{
      if(label===field)label=next[role]||''
      next[role]=field
    }
  }
  if(mode==='single'&&next.length===2&&next.some(name=>byName.get(name)?.kind!=='numeric')){
    return {error:'散点对比需要两个数值字段。查看单字段分布时可先移除对比字段。'}
  }
  if(label){
    const column=byName.get(label)
    if(!column||column.unique+(column.missing>0?1:0)>30)return {error:'分组标签最多支持 30 类（包含缺失值）'}
  }
  return {axes:next,group:label}
}
