import {initialSplit,splitPayload,splitError} from './splitting.js'

export const conditionOperators={eq:'等于',ne:'不等于',in:'属于集合',not_in:'不属于集合',gt:'大于',ge:'大于等于',lt:'小于',le:'小于等于',between:'介于（含端点）',contains:'包含文本',is_null:'为空',not_null:'不为空'}
export const actionTypes={dataset:'保存为数据集',condition:'继续判断条件',split:'继续划分样本',exclude:'排除样本'}
export const emptyCondition=()=>({type:'group',op:'AND',children:[{type:'leaf',field:'',operator:'eq',value:null}]})
export const datasetNode=name=>({type:'dataset',name})
export function splitNode(dataset){
  const form=initialSplit(dataset)
  return {type:'split',form,children:form.parts.map(p=>datasetNode(p.name))}
}
export function actionNode(type,dataset,name=''){
  const prefix=(dataset?.name||'数据集').slice(0,115)
  if(type==='dataset')return datasetNode(name||`${prefix}_结果集`)
  if(type==='exclude')return {type,reason:''}
  if(type==='split')return splitNode(dataset)
  return {type:'condition',rules:[{condition:emptyCondition(),action:datasetNode(`${prefix}_条件留出集`)}],otherwise:datasetNode(`${prefix}_其他样本`)}
}
export function combinationTemplate(dataset){
  const prefix=(dataset?.name||'数据集').slice(0,115),root=actionNode('condition',dataset)
  const field=dataset?.columns.find(c=>/group|分组|渠道/i.test(c.name))?.name||''
  root.rules[0].condition.children[0].field=field
  root.rules[0].action.name=`${prefix}_指定分组留出`
  const random=splitNode(dataset)
  random.form.parts=[{name:'训练样本',percent:80},{name:'测试样本',percent:20}]
  random.children=[datasetNode(`${prefix}_训练集`),datasetNode(`${prefix}_测试集`)]
  const time=splitNode(dataset)
  Object.assign(time.form,{method:'time',time_mode:'cutoff',cutoffs:[''],parts:[{name:'建模样本',percent:80},{name:'时间外样本',percent:20}]})
  time.children=[random,datasetNode(`${prefix}_OOT`)]
  root.otherwise=time
  return root
}
export function serializeTree(node){
  if(node.type==='dataset')return {type:node.type,name:node.name.trim()}
  if(node.type==='exclude')return {type:node.type,reason:node.reason.trim()}
  if(node.type==='condition'){
    // 条件分支没有 parts.name，需要单独保留用户为中间分组填写的名称。
    function branch(child,defaultName){
      const saved=serializeTree(child),label=child.label?.trim()
      if(['split','condition'].includes(child.type)&&label&&label!==defaultName)saved.label=label
      return saved
    }
    return {type:node.type,rules:node.rules.map((r,i)=>({condition:JSON.parse(JSON.stringify(r.condition)),action:branch(r.action,`条件 ${i+1} 命中`)})),otherwise:branch(node.otherwise,'其余样本')}
  }
  const {expected_revision,...config}=splitPayload(node.form,1)
  return {type:'split',config,children:node.children.map(serializeTree)}
}
export function leafNames(node){
  if(node.type==='dataset')return [node.name.trim()]
  if(node.type==='exclude')return []
  return (node.type==='condition'?[...node.rules.map(r=>r.action),node.otherwise]:node.children).flatMap(leafNames)
}
export function treeError(root,existing=[]){
  let count=0
  function visit(node,depth,condition=false){
    if(++count>100||depth>8)return '条件树最多支持 100 个节点、8 层嵌套'
    if(condition){
      if(node.type==='group'){
        if(!node.children.length)return '条件组至少需要一个条件'
        for(const child of node.children){const error=visit(child,depth+1,true);if(error)return error}
      }else{
        if(!node.field)return '请选择条件字段'
        if(!['is_null','not_null'].includes(node.operator)){
          if(['in','not_in','between'].includes(node.operator)){
            if(!Array.isArray(node.value)||!node.value.length||node.value.length>200||(node.operator==='between'&&(node.value.length!==2||node.value.some(v=>v===''||v===null))))return '集合需要 1–200 个取值，区间需填写两个端点'
          }else if(node.value===null||node.value===undefined)return '请填写条件比较值'
        }
      }
      return ''
    }
    if(node.type==='dataset')return !node.name.trim()||node.name.trim().length>150?'每个结果名称需要 1 到 150 个字符':''
    if(node.type==='exclude')return !node.reason.trim()?'请填写排除原因':''
    if(node.type==='split'){
      const error=splitError(node.form)
      if(error)return error
      for(const child of node.children){const error=visit(child,depth+1);if(error)return error}
    }else{
      for(const rule of node.rules){const error=visit(rule.condition,depth+1,true)||visit(rule.action,depth+1);if(error)return error}
      return visit(node.otherwise,depth+1)
    }
    return ''
  }
  const error=visit(root,1)
  if(error)return error
  const names=leafNames(root)
  if(!names.length||names.length>30)return '请配置 1 到 30 个结果数据集'
  if(new Set(names).size!==names.length)return '所有叶子数据集名称必须互不重复'
  if(names.some(name=>existing.includes(name)))return '结果名称与已有数据集重复，请修改名称'
  return ''
}
