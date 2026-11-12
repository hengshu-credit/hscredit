// fx 配置只描述表达式结构；实际表达式编译与计算由服务端负责。
export const fieldNode = name => ({kind:'FIELD',name:name||''})
export const emptyNode = () => ({kind:'EMPTY'})
export const literalNode = (value=0,value_type='NUMBER') => ({kind:'LITERAL',value_type,value})
export const operationNode = node => ({kind:'OPERATION',terms:[{operand:node||fieldNode()}]})
export function parseNumericLiteral(text){
  if(String(text).trim()==='')return null
  const value=Number(text)
  return Number.isFinite(value)?value:null
}
export function functionNode(name,catalog){
  return {kind:'FUNCTION',name,args:(catalog.find(f=>f.name===name)?.args||[]).map(()=>fieldNode())}
}
export function changeNodeKind(kind,fields,catalog){
  if(kind==='EMPTY')return emptyNode()
  if(kind==='UNARY')return {kind:'UNARY',operator:'-',operand:fieldNode(fields[0]?.name)}
  if(kind==='LITERAL')return literalNode()
  if(kind==='FUNCTION')return functionNode('where',catalog)
  if(kind==='OPERATION')return operationNode(fieldNode(fields[0]?.name))
  return fieldNode(fields[0]?.name)
}
export function removeOperationTerm(node,index){
  const terms=node.terms.filter((_,i)=>i!==index).map(t=>structuredClone(t))
  if(terms.length)delete terms[0].operator
  return {...node,terms}
}
export function switchRuleMode(rule,mode,fields){
  const targets={...rule._targets,[rule.mode]:rule.target}
  const remembered=targets[mode]
  const target=remembered??(mode==='replace'&&!fields.some(f=>f.name===rule.target)?'':rule.target)
  return {...rule,mode,target,_targets:targets}
}
export function configuredRules(rules){return rules.map(({mode,target,expression})=>({mode,target,expression}))}
export function expressionComplete(node){
  if(!node||node.kind==='EMPTY')return false
  if(node.kind==='FIELD')return !!node.name
  if(node.kind==='LITERAL')return node.value_type!=='NUMBER'||(typeof node.value==='number'&&Number.isFinite(node.value))
  if(node.kind==='UNARY')return expressionComplete(node.operand)
  if(node.kind==='FUNCTION')return node.args.every(expressionComplete)
  return node.kind==='OPERATION'&&node.terms.length>0&&node.terms.every(t=>expressionComplete(t.operand))
}
export function replaceNode(root,path,value){
  if(!path.length)return value
  const copy=JSON.parse(JSON.stringify(root));let parent=copy
  for(const key of path.slice(0,-1))parent=parent[key]
  parent[path.at(-1)]=value
  return copy
}
export function fieldKind(column){
  if(/bool/i.test(column.dtype))return 'boolean'
  if(column.kind==='numeric')return 'numeric'
  if(column.kind==='date')return 'date'
  return 'string'
}
export function inferredType(node,fields){
  if(node?.kind==='UNARY')return node.operator==='not'?'boolean':inferredType(node.operand,fields)
  if(node?.kind==='FIELD')return fields.find(f=>f.name===node.name)?.valueType||'unknown'
  if(node?.kind==='LITERAL')return {NUMBER:'numeric',STRING:'string',BOOLEAN:'boolean',MISSING:'numeric'}[node.value_type]||'unknown'
  if(node?.kind==='FUNCTION')return ['isnan','isfinite'].includes(node.name)?'boolean':node.name==='where'?inferredType(node.args?.[1],fields):'numeric'
  if(node?.kind==='OPERATION'){
    if(node.terms?.some(t=>['==','!=','>','>=','<','<=','and','or'].includes(t.operator)))return 'boolean'
    return inferredType(node.terms?.[0]?.operand,fields)
  }
  return 'unknown'
}
export function fieldsBefore(columns,rules,index){
  const fields=columns.map(c=>({...c,valueType:fieldKind(c)}))
  for(const rule of rules.slice(0,index)){
    if(!rule.target)continue
    const valueType=inferredType(rule.expression,fields), found=fields.find(f=>f.name===rule.target)
    const derived={name:rule.target,dtype:'前序步骤结果',valueType,kind:valueType==='numeric'?'numeric':'category',values:[],derived:true}
    if(found)Object.assign(found,derived);else fields.push(derived)
  }
  return fields
}
export function describeNode(node){
  if(!node||node.kind==='EMPTY')return ''
  if(node.kind==='FIELD')return node.name?`col(${JSON.stringify(node.name)})`:''
  if(node.kind==='LITERAL')return node.value_type==='MISSING'?'nan':node.value_type==='BOOLEAN'?(node.value?'True':'False'):node.value_type==='STRING'?JSON.stringify(node.value):String(node.value??'')
  if(node.kind==='UNARY')return `(${node.operator==='not'?'not ':node.operator}${describeNode(node.operand)})`
  if(node.kind==='FUNCTION')return `${node.name}(${(node.args||[]).map(describeNode).join(', ')})`
  if(node.kind==='OPERATION')return node.terms?.length?'('+node.terms.map((t,i)=>(i?`${t.operator} `:'')+describeNode(t.operand)).join(' ')+')':''
  return '选择内容'
}
