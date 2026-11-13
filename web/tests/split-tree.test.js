import test from 'node:test'
import assert from 'node:assert/strict'
import {combinationTemplate,serializeTree,treeError,leafNames,actionNode,emptyCondition} from '../src/split-tree.js'
const dataset={name:'贷款样本',columns:[{name:'GROUP',kind:'category'},{name:'日期',kind:'date'}]}

function ready(){const tree=combinationTemplate(dataset);tree.rules[0].condition.children[0].value='留出';tree.otherwise.form.cutoffs=['2025-03-01'];return tree}
test('组合模板生成条件留出、时间外和随机训练测试四个叶子',()=>{
  const tree=ready(),payload=serializeTree(tree)
  assert.equal(treeError(tree),'')
  assert.deepEqual(leafNames(tree),['贷款样本_指定分组留出','贷款样本_训练集','贷款样本_测试集','贷款样本_OOT'])
  assert.equal(payload.rules[0].condition.children[0].field,'GROUP')
  assert.equal(payload.otherwise.config.method,'time')
  assert.equal(payload.otherwise.children[0].config.method,'random')
  assert.deepEqual(payload.otherwise.children[0].config.parts.map(p=>p.ratio),[.8,.2])
  assert.equal('expected_revision' in payload.otherwise.config,false)
  assert.equal('form' in payload.otherwise,false)
})
test('校验包含嵌套比例、全树重名、已有名称和排除原因',()=>{
  const tree=ready()
  tree.otherwise.children[0].form.parts[0].percent=50
  assert.match(treeError(tree),/100%/)
  tree.otherwise.children[0].form.parts[0].percent=80
  tree.otherwise.children[1].name=tree.rules[0].action.name
  assert.match(treeError(tree),/互不重复/)
  tree.otherwise.children[1].name='时间外'
  assert.match(treeError(tree,['时间外']),/已有数据集/)
  tree.otherwise.children[1]={type:'exclude',reason:''}
  assert.match(treeError(tree),/排除原因/)
})
test('条件且或嵌套和操作变化会进入序列化配置',()=>{
  const tree=ready(),group=emptyCondition()
  group.op='OR';group.children=[{type:'leaf',field:'金额',operator:'between',value:[10,100]}]
  tree.rules[0].condition.children.push(group)
  const before=JSON.stringify(serializeTree(tree))
  group.children[0].value[1]=200
  assert.notEqual(JSON.stringify(serializeTree(tree)),before)
  assert.equal(treeError(tree),'')
  group.children[0].value=[]
  assert.match(treeError(tree),/端点/)
})
test('空条件、空名称、过深嵌套不能预览',()=>{
  const tree=actionNode('condition',dataset)
  assert.match(treeError(tree),/条件字段/)
  const value=ready()
  value.otherwise.children[0].children[0].name=' '
  assert.match(treeError(value),/名称/)
  let deep=actionNode('dataset',dataset,'结尾')
  for(let i=0;i<9;i++){const parent=ready();parent.otherwise=deep;deep=parent}
  assert.match(treeError(deep),/嵌套/)
})
