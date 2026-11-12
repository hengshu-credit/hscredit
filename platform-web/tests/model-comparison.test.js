import test from 'node:test'
import assert from 'node:assert/strict'
import {modelParameterSpecs,selectedModelParameters} from '../src/model-options.js'

test('多模型对比按模型标识合并默认值和独立配置，只提交当前选中的模型',()=>{
  const catalog={
    random_forest:{params:{n_estimators:100,max_depth:6}},
    extra_trees:{params:{n_estimators:100,max_depth:6}},
  }
  const overrides={random_forest:{max_depth:3},extra_trees:{n_estimators:50},logistic:{C:2}}
  const params=selectedModelParameters(['extra_trees','random_forest'],catalog,overrides)
  assert.deepEqual(params,{extra_trees:{n_estimators:50,max_depth:6},random_forest:{n_estimators:100,max_depth:3}})
  assert.deepEqual(Object.keys(params),['extra_trees','random_forest'])
  params.extra_trees.max_depth=9
  assert.equal(catalog.extra_trees.params.max_depth,6)
  assert.equal(params.random_forest.max_depth,3)
  assert.deepEqual(selectedModelParameters([],catalog,overrides),{})
})

test('单模型行的编辑器只包含该模型支持的参数和可选值',()=>{
  const specs=modelParameterSpecs({params:{C:1,kernel:'rbf'},choices:{kernel:['rbf','linear']}})
  assert.deepEqual(specs.map(s=>s.key),['C','kernel'])
  assert.deepEqual(specs[1].choices.map(c=>c.value),['rbf','linear'])
  assert.deepEqual(modelParameterSpecs({params:{}}),[])
})
