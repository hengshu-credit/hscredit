import test from 'node:test'
import assert from 'node:assert/strict'
import {modelParametersSchema, categoryGroupsSchema, requestSchema, taskConfigSchema} from '../src/editor-context.js'
import {expressionSuggestions} from '../src/expression-suggestions.js'

test('搜索空间只包含已选模型及其独立的参数、允许值与推荐值', () => {
  const catalog = {
    logistic: {name: '逻辑回归', params: {C: 1}, search: {C: [0.1, 1]}, choices: {}},
    svm: {name: '支持向量机', params: {kernel: 'rbf'}, search: {kernel: ['rbf']}, choices: {kernel: ['rbf','linear']}},
  }
  const schema = selected => modelParametersSchema(selected, catalog)
  const both = schema(['logistic','svm']).properties
  assert.deepEqual(Object.keys(both.logistic.properties), ['C'])
  assert.deepEqual(both.logistic.properties.C.default, [0.1,1])
  assert.deepEqual(both.svm.properties.kernel.items.enum, ['rbf','linear'])
  assert.deepEqual(Object.keys(schema(['svm']).properties), ['svm'])
  assert.deepEqual(schema([]).properties, {})
})

test('类别组提示保留原始值类型，服务请求只在记录内提示当前服务字段', () => {
  const field = {values:[{value:0},{value:false},{value:'0'},{value:null}]}
  assert.deepEqual(categoryGroupsSchema(field).items.items.examples, [0,false,'0'])
  assert.deepEqual(categoryGroupsSchema().items.items.examples, [])
  assert.deepEqual(Object.keys(requestSchema(['评分']).properties.records.items.properties), ['评分'])
})

const fields = [{name: '客户年龄'}, {name:'税后 "收入"', derived:true}]
function complete(source, context, label) {
  const offset = source.indexOf('|'), text = source.replace('|','')
  const completion = expressionSuggestions(text,offset,context)
  const item = completion?.items.find(i=>i.label===label)
  return item && text.slice(0,completion.start) + item.insertText + text.slice(completion.end)
}
test('字段加工补全前序字段、col 参数与函数，字符串常量不参与', () => {
  const ctx = {fields, processing:true, functions:[{name:'abs', label:'绝对值', args:['数值']}]}
  assert.equal(complete('客|户 + 2',ctx,fields[0].name),'col("客户年龄") + 2')
  assert.equal(complete('col("税|后")',ctx,fields[1].name),'col('+JSON.stringify(fields[1].name)+')')
  assert.equal(complete('col(|)',ctx,fields[0].name),'col("客户年龄")')
  assert.equal(complete('ab|',ctx,'abs'),'abs($1)')
  assert.equal(complete('"客|"',ctx,fields[0].name),undefined)
})

test('手写规则 JSON 只在 expression 字符串补全，替换局部并保持 JSON 转义', () => {
  const ctx = {fields,embedded:true}
  const json = expression => JSON.stringify([{name:'示例',expression}])
  const result = complete(json('客户年龄 > 18 and 税| > 0'),ctx,fields[1].name)
  assert.equal(JSON.parse(result)[0].expression,'客户年龄 > 18 and `税后 "收入"` > 0')
  assert.equal(complete(json('客户年龄 == "A|B"'),ctx,fields[1].name),undefined)
  assert.equal(complete('[{"name":"客|"}]',ctx,fields[1].name),undefined)
  assert.equal(complete('客户年龄.is|',{fields},'isna'),'客户年龄.isna($0)')
})

test('完整任务配置按绑定提供字段与模型候选，普通参数填标量，搜索参数填数组', () => {
  const catalog = {forest:{available:true,name:'随机森林',params:{max_depth:6},choices:{},search:{max_depth:[3,5,8]}}}
  const {properties} = taskConfigSchema({kind:'model',cv:4}, fields, [{id:'sample'}], catalog, ['forest'])
  assert.deepEqual(properties.features.items.examples, fields.map(f=>f.name))
  assert.equal(properties.model_params.properties.forest.properties.max_depth.default, 6)
  assert.deepEqual(properties.search_spaces.properties.forest.properties.max_depth.default, [3,5,8])
  const ctx = {embedded:true,ruleProperty:'selected_rules',fields}
  const result = complete('{"selected_rules":[{"expression":"客| > 1"}]}',ctx,'客户年龄')
  assert.equal(JSON.parse(result).selected_rules[0].expression,'客户年龄 > 1')
})
