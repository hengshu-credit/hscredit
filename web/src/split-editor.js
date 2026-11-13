// 样本编辑器：导航与命名仅用于界面，执行仍使用原有划分树契约。
import {datasetNode,emptyCondition} from './split-tree.js'
import {initialSplit,splitMethods} from './splitting.js'

export const editorMethods={condition:{label:'按条件',hint:'按字段条件分出不同样本'},random:{label:'按比例随机',hint:'随机分配，可保持标签比例'},time:{label:'按时间',hint:'按日期边界或时间顺序拆分'},group:{label:'按分组',hint:'同一客户或分组放在同一份'},oot:{label:'留出时间外样本',hint:'指定时间范围作为时间外集'}}
export function uniqueSampleName(base,used=[]){
  const prefix=(base||'样本').slice(0,140),names=new Set(used)
  let name=prefix,index=2
  while(names.has(name))name=`${prefix}_${index++}`
  return name
}
export function sampleRows(root){
  const rows=[]
  function visit(node,path=[],parent=null,index=0,edge=''){
    const name=node.type==='dataset'?node.name:node.label||edge||'全部样本'
    const parentRow=rows.find(row=>row.node===parent),position=index==='otherwise'?parent.rules.length+1:index+1
    rows.push({node,path,key:JSON.stringify(path),parent,index,depth:path.length,name,edge,number:parent?(parentRow.number?`${parentRow.number}.${position}`:String(position)):'',status:node.type==='dataset'?'待保存':node.type==='exclude'?(node.removed?'已移除':'已排除'):node.type==='condition'?'按条件':splitMethods[node.form.method]})
    if(node.type==='split')node.children.forEach((child,i)=>visit(child,[...path,i],node,i,node.form.parts[i].name))
    if(node.type==='condition'){
      node.rules.forEach((rule,i)=>visit(rule.action,[...path,i],node,i,`条件 ${i+1} 命中`))
      visit(node.otherwise,[...path,'otherwise'],node,'otherwise','其余样本')
    }
  }
  visit(root)
  return rows
}
export function renameSample(row,name){
  if(row.node.type==='dataset')row.node.name=name
  else row.node.label=name
  if(row.parent?.type==='split')row.parent.form.parts[row.index].name=name
  if(row.node.removed){row.node.reason=`已移除结果：${name}`.slice(0,150);if(row.node.removed.type==='dataset')row.node.removed.name=name;else row.node.removed.label=name}
}
export function sampleHierarchy(root){
  const rows=sampleRows(root),items=new Map(rows.map(row=>[row.key,{...row,children:[]}]))
  for(const row of rows.slice(1))items.get(JSON.stringify(row.path.slice(0,-1))).children.push(items.get(row.key))
  return [items.get('[]')]
}
export function sampleAncestors(root,path){return sampleRows(root).filter(row=>row.path.length<path.length&&row.path.every((value,i)=>value===path[i]))}
export function sampleSource(row){
  const parent=row.parent
  if(!parent)return '源数据的全部样本'
  if(parent.type==='condition')return row.index==='otherwise'?'未命中以上条件的其余样本':`第 ${row.index+1} 条条件命中的样本`
  const form=parent.form,index=row.index
  if(form.method==='oot')return index===0?`${form.oot_start||'起始时间'}之前`:`${form.oot_start||'起始时间'}至${form.oot_end||'后续全部时间'}`
  if(form.method==='time'&&form.time_mode==='cutoff')return index===0?`早于 ${form.cutoffs[0]||'待设置边界'}`:index===form.parts.length-1?`${form.cutoffs[index-1]||'待设置边界'} 起（含）`:`${form.cutoffs[index-1]||'待设置边界'}（含）至 ${form.cutoffs[index]||'待设置边界'}（不含）`
  return `占上级${form.method==='group'?'分组数':'样本'} ${form.parts[index].percent??'—'}%`
}
export function dismissSample(root,path){
  const row=sampleRows(root).find(item=>item.key===JSON.stringify(path))
  if(!row?.parent||row.node.type==='exclude')return root
  // 保留原划分槽位，只取消该子树的输出，不把这些样本重新分给其他结果。
  return replaceSample(root,path,{type:'exclude',label:row.name,reason:`已移除结果：${row.name}`.slice(0,150),removed:JSON.parse(JSON.stringify(row.node))})
}
export function restoreSample(root,path){
  const row=sampleRows(root).find(item=>item.key===JSON.stringify(path))
  if(!row?.node.removed)return root
  return replaceSample(root,path,row.node.removed)
}
export function makeSampleSplit(method,dataset,name,used=[]){
  const names=[],next=suffix=>{const value=uniqueSampleName(`${name}_${suffix}`,[...used,...names]);names.push(value);return value}
  if(method==='condition')return {type:'condition',label:name,rules:[{condition:emptyCondition(),action:datasetNode(next('符合条件'))}],otherwise:datasetNode(next('其余样本'))}
  const form=initialSplit(dataset)
  form.method=method
  form.time_mode=method==='time'?'cutoff':'ratio'
  form.cutoffs=['']
  form.parts=(method==='oot'?['开发集','时间外集']:['样本1','样本2']).map(suffix=>({name:next(suffix),percent:50}))
  return {type:'split',label:name,form,children:form.parts.map(p=>datasetNode(p.name))}
}
export function appendSample(node,used=[]){
  const name=uniqueSampleName(`${node.label||'样本'}_样本${node.type==='split'?node.children.length+1:node.rules.length+1}`,used)
  if(node.type==='condition')node.rules.push({condition:emptyCondition(),action:datasetNode(name)})
  else{
    // 等比分配只是新增结果时的初始值，用户仍可逐份调整。
    node.form.parts.push({name,percent:0});node.children.push(datasetNode(name));balanceSamples(node)
    node.form.cutoffs.push('')
  }
}
export function balanceSamples(node){
  const share=Math.floor(10000/node.form.parts.length)/100
  node.form.parts.forEach((part,i)=>{part.percent=i===node.form.parts.length-1?Number((100-share*i).toFixed(2)):share})
}
export function removeSample(node,index){
  if(node.type==='condition')node.rules.splice(index,1)
  else{node.children.splice(index,1);node.form.parts.splice(index,1);node.form.cutoffs.splice(Math.max(0,index-1),1);balanceSamples(node)}
}
export function replaceSample(root,path,replacement){
  if(!path.length)return replacement
  const parent=sampleRows(root).find(row=>row.key===JSON.stringify(path.slice(0,-1))).node,index=path.at(-1)
  if(parent.type==='split'){parent.children[index]=replacement;if(replacement.type==='dataset')parent.form.parts[index].name=replacement.name}
  else if(index==='otherwise')parent.otherwise=replacement
  else parent.rules[index].action=replacement
  return root
}
export function availableNames(root,existing=[]){return [...existing,...sampleRows(root).flatMap(row=>[row.name,...(row.node.removed?availableNames(row.node.removed):[])])]}

// 单次划分和组合划分都还原为同一个可编辑的树。
export function restoreSplitConfig(config,dataset,members=[]){
  const names = new Map(members.map(d=>[d.splitting.output_name||d.name,d.name]))
  const byPath=new Map(members.filter(d=>d.splitting.path).map(d=>[d.splitting.path,d.name]))
  function restore(node,label,path){
    label=node.label||label
    if(node.type==='dataset')return datasetNode(byPath.get(path)||names.get(node.name)||node.name)
    if(node.type==='exclude')return {...node,label}
    if(node.type==='condition'){
      return {type:'condition',label,rules:node.rules.map((rule,i)=>({
        condition:JSON.parse(JSON.stringify(rule.condition)),
        action:restore(rule.action,`条件 ${i+1} 命中`,`${path} / 条件${i+1} / 动作`)
      })),otherwise:restore(node.otherwise,'其余样本',path+' / 其他')}
    }
    const saved=node.config,form={...initialSplit(dataset),...saved}
    form.cutoffs=[...(saved.cutoffs||[])]
    const children=node.children.map((child,i)=>restore(child,saved.parts[i].name,`${path} / 划分${i+1}（${saved.parts[i].name.trim()}）`))
    form.parts=saved.parts.map((p,i)=>({
      name:node.children[i].name===p.name?children[i].name:p.name,
      percent:p.ratio==null?100/saved.parts.length:Number((p.ratio*100).toFixed(10))
    }))
    return {type:'split',label,form,children}
  }
  return restore(config.tree||{type:'split',config,children:config.parts.map(p=>datasetNode(p.name))},dataset.name,'全部样本')
}

function compareSplitOutputs(a,b){
  const left=a.splitting,right=b.splitting
  if(left.plan_id===right.plan_id)return left.part_index-right.part_index
  return (right.at||'').localeCompare(left.at||'')||left.plan_id.localeCompare(right.plan_id)
}

export function datasetPlans(datasets,sourceId){
  const plans=new Map()
  for(const child of datasets.filter(d=>d.parent_id===sourceId&&d.splitting?.plan_id).sort(compareSplitOutputs)){
    const id=child.splitting.plan_id
    if(!plans.has(id))plans.set(id,[])
    plans.get(id).push(child)
  }
  return plans
}

export function datasetHierarchy(datasets){
  const nodes=new Map(datasets.map(d=>[d.id,{...d,children:[]}]))
  const roots=[]
  for(const node of nodes.values()){
    const parent=nodes.get(node.parent_id)
    if(parent)parent.children.push(node)
    else roots.push(node)
  }
  for(const node of nodes.values()){
    node.children.sort(compareSplitOutputs)
  }
  return roots
}

// 按配置展示统计和操作；方案标题与未迁移的分支只用于导航。
export function assetHierarchy(roots){
  function asset(row){
    if(!row.children.length)return row
    const plans=datasetPlans(row.children,row.id)
    const children=[]
    for(const [id,outputs] of plans){
      const members=outputs.map(asset)
      const config=members[0].splitting.config
      const prefix=`group:${row.id}:${id}`
      const tree=config?.tree
      const branch=tree&&['condition','split'].includes(tree.type)?groupOutputs(tree,members,prefix):members
      if(plans.size===1)children.push(...branch)
      else{
        const method=config?.tree?.config?.method||config?.method
        const label=config?.tree?.type==='condition'?'按条件':splitMethods[method]
        children.push({id:prefix,name:`方案 ${children.length+1}${label?' · '+label:''}`,isGroup:true,children:branch})
      }
    }
    return {...row,children}
  }

  function groupOutputs(tree,members,prefix){
    const leaves=members.filter(d=>!d.group_path)
    const groupBySlot=new Map(members.filter(d=>d.group_path).map(d=>[JSON.stringify(d.group_path),d]))
    const byName=new Map(leaves.map(d=>[d.splitting.output_name||d.name,d]))
    // 旧版实体子集在重命名后仍保有原始划分路径。
    const byPath=new Map(members.filter(d=>d.splitting.path).map(d=>[d.splitting.path,d]))
    const used=new Set()
    function visit(node,name,slot,path){
      if(node.type==='dataset'){
        const row=byPath.get(path)||byName.get(node.name.trim())
        if(!row||used.has(row.id))return null
        used.add(row.id)
        return row
      }
      if(node.type==='exclude')return null
      let children
      if(node.type==='condition'){
        children=node.rules.map((rule,i)=>visit(rule.action,`条件 ${i+1} 命中`,[...slot,i],`${path} / 条件${i+1} / 动作`))
        children.push(visit(node.otherwise,'其余样本',[...slot,'otherwise'],path+' / 其他'))
      }else{
        children=node.children.map((child,i)=>{
          const part=node.config.parts[i].name.trim()
          return visit(child,part,[...slot,i],`${path} / 划分${i+1}（${part}）`)
        })
      }
      children=children.filter(Boolean)
      const key=JSON.stringify(slot),group=groupBySlot.get(key)
      if(group)return {...group,children}
      return children.length?{id:prefix+key,name:node.label||name,isGroup:true,children}:null
    }
    const groups=visit(tree,'全部样本',[],'全部样本')
    // 仅展示有实际输出的路径；无法匹配旧配置的资产仍保留在所属方案。
    return [...(groups?.children||[]),...leaves.filter(row=>!used.has(row.id))]
  }
  return roots.map(asset)
}
