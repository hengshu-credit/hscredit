import assert from 'node:assert/strict'
import test from 'node:test'
import {filterRows, summaryFields, useTableFilters} from '../src/table-filters.js'

test('添加字段到首个值生效之前保留原筛选，移除某一字段保留其他条件', () => {
  const {drafts, conditions} = useTableFilters()
  drafts.value = [{key:'状态', values:['成功','失败']}]
  const original = conditions.value
  drafts.value = [...drafts.value, {key:'', values:[]}]
  assert.equal(conditions.value, original)
  drafts.value = [drafts.value[0], {key:'版本', values:[]}]
  assert.equal(conditions.value, original)
  drafts.value = [drafts.value[0], {key:'版本', values:[0]}]
  assert.deepEqual(conditions.value.map(row => row.values), [['成功','失败'],[0]])
  drafts.value = drafts.value.slice(1)
  assert.deepEqual(conditions.value.map(row => row.key), ['版本'])
})

test('字段概况按分组列的真实键筛选，区分零、空串、缺失和同名文本', () => {
  const fields = summaryFields([{label:'详情', children:[{key:'详情 / 众数', label:'众数'}]}])
  const key = fields[0].key
  const rows = [0, null, '', '缺失值', '0'].map(value => ({[key]:value}))
  assert.deepEqual(filterRows(rows, fields, [{key, values:[0, null]}]), rows.slice(0, 2))
})

test('任务筛选按显示值跨字段组合，轮询更新后重新匹配', () => {
  const fields = [{key:'status'}, {key:'version', read:job => job.revision || 1}]
  const rows = [{status:'成功'}, {status:'失败', revision:1}, {status:'成功', revision:2}, {status:'运行中'}]
  const conditions = [{key:'status', values:['成功','失败']}, {key:'version', values:[1]}]
  assert.deepEqual(filterRows(rows, fields, conditions), rows.slice(0, 2))
  rows[3].status = '成功'
  assert.deepEqual(filterRows(rows, fields, conditions), [rows[0],rows[1],rows[3]])
  assert.equal(filterRows(rows, fields, []), rows)
})
