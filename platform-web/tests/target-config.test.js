import test from 'node:test'
import assert from 'node:assert/strict'
import {effectScope, ref} from 'vue'
import {copyTargetDraft,parseThresholds, useTargetConfig} from '../src/target-config.js'

test('字典顺序决定默认主标签，显式切换与移除主标签后回退', () => {
  const scope = effectScope()
  const dataset = {id:'d', columns:[{name:'MOB1', kind:'numeric'}, {name:'MOB3', kind:'numeric'}]}
  const config = scope.run(() => useTargetConfig(() => dataset, 'training'))
  config.draft.mode = 'dict'
  const first = config.draft.entries[0], second = config.newEntry()
  Object.assign(first, {name:'30', field:'MOB3', operator:'>', value:30})
  Object.assign(second, {name:'7', field:'MOB1', operator:'>=', value:7})
  config.draft.entries.push(second)
  assert.equal(config.error, '')
  assert.equal(config.primary, '30')
  assert.deepEqual(config.payload.target_order, ['30','7'])
  assert.deepEqual(config.payload.target['7'], {field:'MOB1', operator:'>=', value:7})
  assert.deepEqual(config.sources, ['MOB3','MOB1'])
  config.primary = '7'
  assert.equal(config.payload.primary_index, 1)
  config.draft.entries.pop()
  assert.equal(config.primary, '30')
  config.draft.entries.push({...second, name:'30'})
  assert.match(config.error, /重复/)
  scope.stop()
})

test('阈值支持小数、中英文分隔和按序去重，拒绝非法输入', () => {
  assert.deepEqual(parseThresholds('7，0 7.0, 0.25').dpds, [7, 0, 0.25])
  for (const text of ['', '-1', '7,坏', 'Infinity', '1e500', '0x10', Array.from({length:21}, (_, i) => i).join(',')]) {
    assert.ok(parseThresholds(text).error)
  }
})

test('只有当前来源进入请求，主标签跟随所选组合，换版本重置草稿', () => {
  const dataset = ref({id:'d', revision:1, default_target:'分类', default_overdue:['MOB1','MOB3'],
    columns:[{name:'分类',unique:2,values:[{value:false},{value:true}]}, ...['MOB1','MOB3'].map(name => ({name,kind:'numeric'}))]})
  const scope = effectScope()
  const config = scope.run(() => useTargetConfig(() => dataset.value, 'training'))
  config.draft.positives['分类'] = false
  assert.equal(config.payload.positive_label, false)
  assert.deepEqual(config.sources, ['分类'])
  assert.ok(!('overdue' in config.payload))
  Object.assign(config.draft, {mode:'overdue', text:'7 30', grey:true})
  config.primary = 'MOB3 > 7 天'
  assert.equal(config.payload.primary_index, 2)
  config.draft.operator = '<='
  assert.equal(config.primary, 'MOB3 <= 7 天')
  assert.equal(config.payload.del_grey, false)
  assert.ok(!('target' in config.payload))
  assert.deepEqual(config.sources, ['MOB1','MOB3'])
  config.draft.mode = 'target'
  assert.equal(config.payload.positive_label, false)
  dataset.value = {...dataset.value, revision:2}
  assert.equal(config.payload.positive_label, true)
  assert.equal(config.draft.text, '7')
  scope.stop()
})

test('文本标签要求明确选择坏样本，修改其他标签时保留已选值', () => {
  const scope = effectScope()
  const dataset = {id:'d', columns:[{name:'结果', unique:2, values:[{value:'坏'}, {value:'好'}]}]}
  const config = scope.run(() => useTargetConfig(() => dataset, 'report'))
  assert.ok(config.error)
  config.draft.positives['结果'] = '坏'
  assert.equal(config.error, '')
  assert.equal(config.payload.positive_label, '坏')
  scope.stop()
})
test('弹窗修改保持独立，确认时保留坏样本取值、字典顺序和训练口径',()=>{
  const scope=effectScope()
  const dataset={id:'甲',columns:[{name:'标签',unique:2,values:[{value:0},{value:1}]},{name:'月份',kind:'numeric'}]}
  const saved=scope.run(()=>useTargetConfig(()=>dataset,'training'))
  const edit=scope.run(()=>useTargetConfig(()=>dataset,'training'))
  saved.draft.positives['标签']=0
  copyTargetDraft(edit.draft,saved.draft)
  assert.equal(edit.payload.positive_label,0)
  edit.draft.mode='dict'
  Object.assign(edit.draft.entries[0],{name:'首逾',field:'标签'})
  edit.draft.entries.push({...edit.newEntry(),name:'月逾期',field:'月份',operator:'>',value:7})
  edit.primary='月逾期'
  assert.equal(saved.payload.target,'标签')
  assert.equal(saved.draft.entries.length,1)
  copyTargetDraft(saved.draft,edit.draft)
  assert.equal(saved.payload.primary_index,1)
  assert.deepEqual(saved.payload.target_order,['首逾','月逾期'])
  edit.draft.entries[1].value=90
  assert.equal(saved.payload.target['月逾期'].value,7)
  const next=saved.newEntry()
  assert.ok(!saved.draft.entries.some(item=>item.id===next.id))
  scope.stop()
})
