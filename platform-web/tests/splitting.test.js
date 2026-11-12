import test from 'node:test'
import assert from 'node:assert/strict'
import {initialSplit,splitPayload,splitError} from '../src/splitting.js'

const dataset={name:'贷款数据',columns:[{name:'申请日期',kind:'date'}]}
test('随机配置包含独立名称、比例和固定种子',()=>{
  const form=initialSplit(dataset)
  assert.equal(splitError(form),'')
  assert.deepEqual(splitPayload(form,3),{method:'random',expected_revision:3,parts:[{name:'贷款数据_训练集',ratio:.6},{name:'贷款数据_验证集',ratio:.2},{name:'贷款数据_测试集',ratio:.2}],random_state:42,stratify:null})
})
test('方法切换只提交当前方法的配置',()=>{
  const form=initialSplit(dataset)
  Object.assign(form,{stratify:'标签',method:'group',group_col:'客户'})
  const grouped=splitPayload(form,1)
  assert.equal(grouped.group_col,'客户')
  assert.equal('stratify' in grouped,false)
  Object.assign(form,{method:'time',time_mode:'cutoff',cutoffs:['2025-02-01T00:00:00','2025-03-01T00:00:00','旧边界']})
  assert.equal(splitError(form),'')
  const time=splitPayload(form,1)
  assert.equal(time.cutoffs.length,2)
  assert.equal('ratio' in time.parts[0],false)
  assert.equal('random_state' in time,false)
})
test('名称、比例、边界和分组必填校验',()=>{
  const form=initialSplit(dataset)
  assert.match(splitError(form,[form.parts[0].name]),/已有数据集/)
  form.parts[1].name=' '+form.parts[0].name+' '
  assert.match(splitError(form),/不能重复/)
  form.parts[1].name='验证'
  form.parts[0].percent=30
  assert.match(splitError(form),/100%/)
  form.parts[0].percent=60
  form.method='group'
  assert.match(splitError(form),/分组字段/)
  Object.assign(form,{method:'time',time_mode:'cutoff',cutoffs:['2025-03-01T00:00:00','2025-02-01T00:00:00']})
  assert.match(splitError(form),/从早到晚/)
  form.cutoffs=[]
  assert.match(splitError(form),/时间边界/)
})
test('时间外划分固定为两份且允许不填结束时间',()=>{
  const form=initialSplit(dataset)
  Object.assign(form,{method:'oot',oot_start:'2025-02-01T00:00:00'})
  assert.match(splitError(form),/两个/)
  form.parts=form.parts.slice(0,2)
  assert.equal(splitError(form),'')
  const payload=splitPayload(form,1)
  assert.equal(payload.oot_end,null)
  assert.equal('ratio' in payload.parts[0],false)
})
