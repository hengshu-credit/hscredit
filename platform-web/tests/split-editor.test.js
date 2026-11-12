import test from 'node:test'
import assert from 'node:assert/strict'
import {sampleRows,renameSample,makeSampleSplit,appendSample,removeSample,replaceSample,availableNames,dismissSample,restoreSample,sampleHierarchy,sampleAncestors,sampleSource} from '../src/split-editor.js'
import {serializeTree,treeError,leafNames,datasetNode} from '../src/split-tree.js'
const dataset={name:'贷款',columns:[{name:'日期',kind:'date'}]}
test('不同分支可以分别使用时间和条件，不影响同级样本',()=>{
  let root=makeSampleSplit('random',dataset,'贷款')
  const time=makeSampleSplit('time',dataset,'早期',availableNames(root))
  time.form.cutoffs=['2025-06-01']
  root=replaceSample(root,[0],time)
  const condition=makeSampleSplit('condition',dataset,'客群',availableNames(root))
  condition.rules[0].condition.children=[{type:'leaf',field:'金额',operator:'gt',value:100}]
  root=replaceSample(root,[1],condition)
  const payload=serializeTree(root)
  assert.equal(treeError(root),'')
  assert.equal(payload.config.method,'random')
  assert.equal(payload.children[0].config.method,'time')
  assert.equal(payload.children[1].type,'condition')
  assert.equal('label' in payload,false)
  assert.equal(leafNames(root).length,4)
  assert.deepEqual(sampleRows(root).map(row=>row.path),[[],[0],[0,0],[0,1],[1],[1,0],[1,'otherwise']])
})
test('只编辑一份名称，同时更新分支路径和最终保存名称',()=>{
  const root=makeSampleSplit('time',dataset,'贷款')
  const sibling=JSON.stringify(root.children[1])
  renameSample(sampleRows(root)[1],'开发样本')
  assert.equal(root.form.parts[0].name,'开发样本')
  assert.equal(root.children[0].name,'开发样本')
  assert.equal(JSON.stringify(root.children[1]),sibling)
  replaceSample(root,[0],makeSampleSplit('random',dataset,'开发样本',availableNames(root)))
  renameSample(sampleRows(root)[1],'开发子集')
  assert.equal(root.form.parts[0].name,'开发子集')
  assert.equal(sampleRows(root)[1].name,'开发子集')
  assert.equal(leafNames(root).length,3)
})
test('新增与移除结果保持比例总和、分支数量与名称有效',()=>{
  const root=makeSampleSplit('random',dataset,'贷款')
  for(let i=0;i<8;i++)appendSample(root,availableNames(root))
  assert.equal(treeError(root),'')
  assert.equal(root.children.length,10)
  const removed=root.children[1].name
  removeSample(root,1)
  appendSample(root,availableNames(root))
  assert.equal(treeError(root),'')
  assert.equal(leafNames(root).includes(removed),false)
  assert.equal(new Set(leafNames(root)).size,10)
})
test('其余样本可继续拆分、排除并恢复，其他条件保持不变',()=>{
  let root=makeSampleSplit('condition',dataset,'贷款')
  const original=JSON.stringify(root.rules)
  root=replaceSample(root,['otherwise'],makeSampleSplit('group',dataset,'其余'))
  root.otherwise.form.group_col='客户'
  root=replaceSample(root,['otherwise',1],{type:'exclude',label:'不保留',reason:'范围外'})
  assert.equal(leafNames(root).length,2)
  root=replaceSample(root,['otherwise',1],datasetNode('恢复样本'))
  assert.equal(root.otherwise.form.parts[1].name,'恢复样本')
  assert.equal(JSON.stringify(root.rules),original)
  assert.equal(leafNames(root).length,3)
})
test('初始命名避开已有数据集以及多次拆分的结果名称',()=>{
  const existing=['贷款_样本1','贷款_样本2']
  const root=makeSampleSplit('random',dataset,'贷款',existing)
  assert.equal(treeError(root,existing),'')
  const next=makeSampleSplit('random',dataset,'贷款',availableNames(root,existing))
  assert.ok(leafNames(next).every(name=>!availableNames(root,existing).includes(name)))
})

for(const method of ['random','time','group','oot','condition'])test(`${method} 两份结果也可以移除其中一份，不改变同级输入配置`,()=>{
  let root=makeSampleSplit(method,dataset,'贷款')
  const config=JSON.stringify(root.form??root.rules[0].condition),sibling=JSON.stringify(method==='condition'?root.otherwise:root.children[1])
  const original=serializeTree(root),removed=sampleRows(root)[1].name
  root=dismissSample(root,[0])
  assert.equal(leafNames(root).includes(removed),false)
  assert.equal(leafNames(root).length,1)
  assert.equal(JSON.stringify(root.form??root.rules[0].condition),config)
  assert.equal(JSON.stringify(method==='condition'?root.otherwise:root.children[1]),sibling)
  const payload=serializeTree(root),action=method==='condition'?payload.rules[0].action:payload.children[0]
  assert.deepEqual(action,{type:'exclude',reason:`已移除结果：${removed}`})
  assert.deepEqual(serializeTree(restoreSample(root,[0])),original)
})
test('父级移除包含所有子级，恢复后边界、名称和子级结构一致',()=>{
  let root=makeSampleSplit('random',dataset,'贷款')
  const time=makeSampleSplit('time',dataset,'开发样本',availableNames(root))
  time.form.cutoffs=['2025-06-01']
  replaceSample(root,[0],time)
  const before=serializeTree(root)
  root=dismissSample(root,[0])
  assert.equal(sampleRows(root).length,3)
  assert.equal(leafNames(root).length,1)
  assert.ok(availableNames(root).includes('开发样本_样本1'))
  renameSample(sampleRows(root)[1],'改名后的开发样本')
  root=restoreSample(root,[0])
  assert.equal(root.children[0].label,'改名后的开发样本')
  assert.deepEqual(serializeTree(root).children[0],before.children[0])
})
test('其余样本可独立移除，全部移除时禁止提交，恢复可重新使用',()=>{
  let root=makeSampleSplit('condition',dataset,'贷款')
  root.rules[0].condition.children=[{type:'leaf',field:'标签',operator:'eq',value:1}]
  root=dismissSample(root,['otherwise'])
  assert.equal(treeError(root),'')
  root=dismissSample(root,[0])
  assert.match(treeError(root),/1 到 30/)
  root=restoreSample(root,['otherwise'])
  assert.equal(treeError(root),'')
  const initial=JSON.stringify(root)
  assert.equal(JSON.stringify(dismissSample(root,[])),initial)
})
test('层级编号、祖先路径与来源只描述当前分支，改名后同步',()=>{
  let root=makeSampleSplit('random',dataset,'贷款')
  replaceSample(root,[1],makeSampleSplit('condition',dataset,'客群'))
  replaceSample(root,[1,'otherwise'],makeSampleSplit('time',dataset,'其余'))
  root.children[1].otherwise.form.cutoffs=['2025-06-01']
  const rows=sampleRows(root),leaf=rows.find(row=>row.key==='[1,"otherwise",0]')
  assert.equal(leaf.number,'2.2.1')
  assert.deepEqual(sampleAncestors(root,leaf.path).map(row=>row.key),['[]','[1]','[1,"otherwise"]'])
  assert.equal(sampleSource(leaf),'早于 2025-06-01')
  assert.equal(sampleHierarchy(root)[0].children[1].children[1].children[0].number,'2.2.1')
  renameSample(rows.find(row=>row.key==='[1]'),'新客群')
  assert.equal(sampleAncestors(root,leaf.path)[1].name,'新客群')
})
