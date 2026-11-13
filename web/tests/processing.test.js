import test from 'node:test'
import assert from 'node:assert/strict'
import {emptyNode,fieldNode,literalNode,operationNode,replaceNode,fieldsBefore,describeNode,parseNumericLiteral,removeOperationTerm,switchRuleMode,configuredRules,expressionComplete} from '../src/processing.js'
test('数值常量保留小数、负数和科学计数值，不把空值当作零',()=>{
  assert.equal(parseNumericLiteral('0.0001'),0.0001)
  assert.equal(parseNumericLiteral('-2.5'),-2.5)
  assert.equal(parseNumericLiteral('1e4'),10000)
  assert.equal(parseNumericLiteral(''),null)
  assert.equal(parseNumericLiteral('-'),null)
  assert.equal(parseNumericLiteral('Infinity'),null)
})
test('嵌套选择器只替换所选操作数，保留其他配置',()=>{
  const original=operationNode(fieldNode('金额'))
  original.terms.push({operator:'/',operand:literalNode(10000)})
  const updated=replaceNode(original,['terms',0,'operand'],fieldNode('余额'))
  assert.equal(updated.terms[0].operand.name,'余额')
  assert.equal(updated.terms[1].operand.value,10000)
  assert.equal(original.terms[0].operand.name,'金额')
})
test('字段选项包含前序结果，排除未来结果并保留布尔类型',()=>{
  const columns=[{name:'金额',dtype:'float64',kind:'numeric'},{name:'开关',dtype:'bool',kind:'numeric'}]
  const rules=[{target:'标签',expression:{kind:'OPERATION',terms:[{operand:fieldNode('金额')},{operator:'>',operand:literalNode(10)}]}},{target:'未来字段',expression:literalNode(1)}]
  const fields=fieldsBefore(columns,rules,1)
  assert.deepEqual(fields.map(f=>f.name),['金额','开关','标签'])
  assert.equal(fields.find(f=>f.name==='标签').valueType,'boolean')
  assert.equal(fields.find(f=>f.name==='开关').valueType,'boolean')
  assert.equal(describeNode(rules[0].expression),'(col("金额") > 10)')
})
test('移除外层运算项会删除完整子树，最后一项也能移除',()=>{
  const nested=operationNode(fieldNode('子字段'))
  nested.terms.push({operator:'+',operand:literalNode(10)})
  const root=operationNode(nested)
  const removed=removeOperationTerm(root,0)
  assert.deepEqual(removed.terms,[])
  assert.equal(expressionComplete(removed),false)
  assert.equal(describeNode(removed),'')
  assert.equal(root.terms[0].operand.terms.length,2)
  assert.equal(expressionComplete(emptyNode()),false)
})
test('切换新增与更新保留表达式及各自字段名，不把界面草稿传入后台',()=>{
  const fields=[{name:'金额'}],expression=operationNode(fieldNode('金额'))
  let rule={_id:'field-one',mode:'add',target:'额度万元',expression}
  rule=switchRuleMode(rule,'replace',fields)
  assert.equal(rule.target,'');assert.equal(rule.expression,expression)
  rule.target='金额'
  rule=switchRuleMode(rule,'add',fields)
  assert.equal(rule.target,'额度万元');assert.equal(rule.expression,expression)
  rule=switchRuleMode(rule,'replace',fields)
  assert.equal(rule.target,'金额')
  assert.deepEqual(Object.keys(configuredRules([rule])[0]).sort(),['expression','mode','target'])
})
