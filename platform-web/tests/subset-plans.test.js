import test from 'node:test'
import assert from 'node:assert/strict'
import {assetHierarchy,datasetHierarchy,datasetPlans,restoreSplitConfig} from '../src/split-editor.js'
import {combinationTemplate,serializeTree} from '../src/split-tree.js'

const source={id:'s',name:'原始样本',columns:[{name:'GROUP',kind:'category'},{name:'日期',kind:'date'}]}

test('源数据默认载入最近保存的方案，副本从空白配置开始',()=>{
  const child=(parent_id,plan_id,part_index,at)=>({parent_id,splitting:{plan_id,part_index,at}})
  const rows=[child('s','old',0,'2026-03-10'),child('s','latest',1,'2026-04-01'),child('s','latest',0,'2026-04-01'),child('other','unrelated',0,'2026-05-01')]
  const plans=datasetPlans(rows,'s')
  assert.deepEqual([...plans.keys()],['latest','old'])
  assert.deepEqual(plans.get('latest'),[rows[2],rows[1]])
  assert.equal(datasetPlans(rows,'copy').size,0)
})

test('已有组合划分重新打开后保持有效条件并可调整比例',()=>{
  const tree=combinationTemplate(source)
  tree.rules[0].condition.children[0].value='A'
  tree.otherwise.form.cutoffs=['2025-03-01']
  const saved=serializeTree(tree)
  const restored=restoreSplitConfig({tree:saved},source)
  const reopened=serializeTree(restored)
  assert.deepEqual(reopened,saved)
  assert.deepEqual(reopened.otherwise.config.cutoffs,['2025-03-01'])
  const random=restored.otherwise.children[0]
  assert.deepEqual(random.form.parts.map(p=>p.percent),[80,20])
  random.form.parts[0].percent=75
  random.form.parts[1].percent=25
  assert.deepEqual(serializeTree(restored).otherwise.children[0].config.parts.map(p=>p.ratio),[.75,.25])
  assert.deepEqual(saved.otherwise.children[0].config.parts.map(p=>p.ratio),[.8,.2])
})

test('旧式单次划分还原为可编辑树，并沿用当前子集名称',()=>{
  const saved={method:'group',group_col:'客户',random_state:8,parts:[{name:'训练',ratio:.6},{name:'测试',ratio:.4}]}
  const form=restoreSplitConfig(saved,source,[{name:'外测',splitting:{output_name:'测试'}}])
  assert.equal(form.form.group_col,'客户')
  assert.equal(form.form.random_state,8)
  assert.deepEqual(form.children.map(n=>n.name),['训练','外测'])
})

test('源数据树内按输出顺序展示子集，嵌套子集不重复计入根节点',()=>{
  const make=(id,parent_id,part_index)=>({id,parent_id,splitting:{plan_id:'p',part_index}})
  const roots=datasetHierarchy([make('c','s',1),make('g','c',0),source,make('b','s',0)])
  assert.deepEqual(roots.map(d=>d.id),['s'])
  assert.deepEqual(roots[0].children.map(d=>d.id),['b','c'])
  assert.deepEqual(roots[0].children[1].children.map(d=>d.id),['g'])
})

test('资产表按条件、时间和随机划分的次序显示中间分组，重命名及下级方案沿用实际数据集',()=>{
  const tree=combinationTemplate(source)
  tree.otherwise.label='常规渠道'
  const config={tree:serializeTree(tree)}
  assert.equal(restoreSplitConfig(config,source).otherwise.label,'常规渠道')
  const child=(id,name,part_index,extra={})=>({id,name,parent_id:source.id,splitting:{plan_id:'p',config,part_index,...extra}})
  const members=[
    child('a','渠道留出',0,{output_name:tree.rules[0].action.name}),
    child('b','训练',1,{path:'全部样本 / 其他 / 划分1（建模样本） / 划分1（训练样本）'}),
    child('c',tree.otherwise.children[0].children[1].name,2),
    child('d',tree.otherwise.children[1].name,3),
  ]
  const nested={id:'inner',name:'内训练',parent_id:'b',splitting:{plan_id:'inner-plan',part_index:0}}
  const physical=datasetHierarchy([nested,...members.slice().reverse(),source])
  const [root]=assetHierarchy(physical)
  assert.deepEqual(root.children.map(d=>d.name),['渠道留出','常规渠道'])
  const [model,oot]=root.children[1].children
  assert.equal(model.name,'建模样本')
  assert.equal(oot.id,'d')
  assert.deepEqual(model.children.map(d=>d.id),['b','c'])
  assert.equal(model.children[0].children[0].id,'inner')
  assert.equal(root.children[1].isGroup,true)
  assert.deepEqual(physical[0].children.map(d=>d.id),['a','b','c','d'])
})

test('空分支不显示，多套方案中同名中间分组的节点标识互相独立',()=>{
  const tree=serializeTree(combinationTemplate(source))
  tree.rules[0].action={type:'exclude',reason:'排除'}
  const name=tree.otherwise.children[1].name
  const children=['p1','p2'].map(plan_id=>({id:plan_id,parent_id:source.id,name:plan_id,splitting:{plan_id,part_index:0,config:{tree},output_name:name}}))
  const [root]=assetHierarchy(datasetHierarchy([source,...children]))
  const [one,two]=root.children
  assert.deepEqual([one,two].map(plan=>plan.children.map(group=>[group.name,group.children.map(d=>d.name)])),[
    [['其余样本',['p1']]], [['其余样本',['p2']]]
  ])
  assert.notEqual(one.children[0].id,two.children[0].id)
})

test('中间分组在资产表和分析选择树中使用后端样本标识、版本与真实统计',()=>{
  const config={tree:serializeTree(combinationTemplate(source))}
  const info={plan_id:'p',config}
  const group={id:'group-id',group_path:['otherwise'],splitting:info,parent_id:'s',name:'其余样本',rows:100,revision:3}
  const output={id:'oot-id',parent_id:'s',name:'独立OOT',rows:15,splitting:{...info,part_index:0,output_name:config.tree.otherwise.children[1].name}}
  const [asset]=assetHierarchy(datasetHierarchy([group,source,output]))
  assert.equal(asset.children.length,1)
  const [branch]=asset.children
  assert.equal(branch.id,group.id)
  assert.equal(branch.rows,100)
  assert.equal(branch.revision,3)
  assert.ok(!branch.isGroup)
  assert.deepEqual(branch.children.map(d=>d.id),['oot-id'])
})
