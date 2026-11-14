import test from 'node:test'
import assert from 'node:assert/strict'
import {treeEditPayload} from '../src/tree-edits.js'

const fields = [{name:'评分', kind:'numeric'}, {name:'布尔', kind:'numeric', dtype:'boolean'}]
const split = {action:'split', node:2, feature:'评分', values:'0', categories:''}

test('分裂草稿保留零阈值，拒绝空阈值和失效字段', () => {
  const parse = overrides => treeEditPayload({...split, ...overrides}, 'binary', fields)
  assert.deepEqual(parse({}), {action:'split', node:2, feature:'评分', threshold:0})
  for (const values of ['', ' ', '2,', '0,5', 'Infinity']) assert.throws(() => parse({values}), /阈值/)
  assert.throws(() => parse({feature:'失效字段'}), /分析字段/)
  assert.throws(() => parse({node:null}), /节点编号/)
})

test('多叉分裂只提交所选字段对应的规则，剪枝只提交节点', () => {
  const parse = overrides => treeEditPayload({...split, ...overrides}, 'multiway', fields)
  assert.deepEqual(parse({values:'0， 5'}).thresholds, [0, 5])
  assert.throws(() => parse({values:'2,1'}), /递增/)
  const groups = {feature:'布尔', categories:'[[true],[false,null]]'}
  assert.deepEqual(parse(groups), {action:'split', node:2, feature:'布尔', category_groups:[[true],[false,null]]})
  assert.throws(() => parse({...groups, categories:'[true]'}), /类别组/)
  assert.deepEqual(parse({...groups, action:'prune'}), {action:'prune', node:2})
})
