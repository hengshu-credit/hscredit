import test from 'node:test'
import assert from 'node:assert/strict'
import {effectScope,ref} from 'vue'
import {useSamplePlan} from '../src/sample-plan.js'

test('同模型的跨样本评估只占一项任务，独立测试不会同时随机留出', () => {
  const assets=ref(['训练','测试','OOT','另一训练'].map((id,i)=>({id,revision:i+1})))
  const primary=ref('训练'),context=ref('model'),scope=effectScope()
  const plan=scope.run(()=>useSamplePlan(assets,primary,context))
  assert.deepEqual(plan.payload,{evaluation_datasets:[],test_dataset:null,use_holdout:true,test_size:0.25})
  plan.draft.testMode='external'
  assert.ok(plan.error)
  plan.draft.test='测试'
  plan.draft.evaluations=['OOT']
  assert.deepEqual(plan.runDatasets.map(d=>d.id),['训练'])
  assert.deepEqual(plan.payload,{evaluation_datasets:[{id:'OOT',revision:3}],test_dataset:{id:'测试',revision:2},use_holdout:false})
  plan.draft.batch=['另一训练']
  assert.equal(plan.runDatasets.length,2)
  plan.draft.testMode='train'
  assert.equal(plan.payload.test_dataset,null)
  assert.equal(plan.payload.use_holdout,false)
  context.value='summary'
  plan.draft.psi='OOT'
  assert.deepEqual(plan.payload,{psi_dataset:{id:'OOT',revision:3}})
  assert.equal(plan.runDatasets.length,1)
  context.value='interaction'
  assert.deepEqual(plan.payload,{})
  context.value='model'
  primary.value='OOT'
  assert.deepEqual(plan.payload.evaluation_datasets,[])
  assets.value=assets.value.filter(d=>d.id!=='另一训练')
  assert.deepEqual(plan.runDatasets.map(d=>d.id),['OOT'])
  scope.stop()
})

test('同版本原表和派生子集用于拟合与评估时提示样本重叠',()=>{
  const source={id:'p',name:'订单',revision:3}
  const sub={id:'s',name:'OOT',parent_id:'p',revision:1,splitting:{source_revision:3}}
  const assets=ref([source,sub]),scope=effectScope()
  const plan=scope.run(()=>useSamplePlan(assets,ref('p'),ref('strategy')))
  plan.draft.evaluations=['s']
  assert.deepEqual(plan.sharedRows,['「订单」与「OOT」'])
  assets.value=[{...source,revision:4},sub]
  assert.deepEqual(plan.sharedRows,[])
  scope.stop()
})
