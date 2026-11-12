import test from 'node:test'
import assert from 'node:assert/strict'
import {addFields,fieldNameIndex,fieldTerms,matchFieldNames,resolveFieldTerms,suggestedFieldNames} from '../src/field-selection.js'

test('手工清单按输入次序匹配并去重，保留原字段名，歧义和未匹配项单独返回',()=>{
  const index=fieldNameIndex(['MOB1','MOB3','评分','Tag','TAG','有 空格','a,b'])
  assert.deepEqual(matchFieldNames('mob3\n评分，MOB1\tmob3;tag、未知；Tag',index),{
    matched:['MOB3','评分','MOB1','Tag'],unknown:['未知'],ambiguous:['tag'],
  })
  assert.deepEqual(matchFieldNames('["a,b","有 空格","mob3"]',index).matched,['a,b','有 空格','MOB3'])
  assert.deepEqual(matchFieldNames('a,b',index).matched,['a,b'])
  assert.deepEqual(matchFieldNames("['MOB1', 'mob3']",index).matched,['MOB1','MOB3'])
  assert.deepEqual(matchFieldNames('有 空格\nMOB3 MOB1',index).matched,['有 空格','MOB3','MOB1'])
})

test('选中搜索结果保留原有顺序与筛选外字段，上限按不同字段计数',()=>{
  const selected=['其他','MOB6']
  assert.deepEqual(addFields(selected,['MOB1','MOB3','MOB3','MOB6'],3),{names:['其他','MOB6','MOB1'],overflow:1})
  assert.deepEqual(selected,['其他','MOB6'])
  assert.deepEqual(addFields([],Array.from({length:8600},(_,i)=>String(i))).names.length,8600)
})

test('手工输入片段实时返回原始字段名，完全相同、前缀、包含依次展示',()=>{
  const index=fieldNameIndex(['pre_BASE','BASE_X2','Base','base','BASE_X1','其他'])
  assert.deepEqual(suggestedFieldNames('BASE',index),['Base','base','BASE_X2','BASE_X1','pre_BASE'])
  assert.deepEqual(suggestedFieldNames('base',index).slice(0,2),['base','Base'])
  assert.deepEqual(suggestedFieldNames('X1',index),['BASE_X1'])
  assert.deepEqual(suggestedFieldNames('',index),[])
  assert.deepEqual(suggestedFieldNames('.*',index),[])
})

test('候选勾选只作用于对应输入项，保持清单顺序、去重及字段范围',()=>{
  const index=fieldNameIndex(['MOB6','BASE_X2','BASE_X1','有 空格','a,b'])
  const text='  MOB6\n BASE，a,b  '
  const terms=fieldTerms(text,index)
  assert.deepEqual(terms.map(t=>text.slice(t.start,t.end)),['MOB6','BASE','a','b'])
  const quoted=' ["BASE", "a,b","有 空格", "BASE_X1", "MOB6"]'
  const list=fieldTerms(quoted,index)
  assert.deepEqual(list.map(t=>JSON.parse(quoted.slice(t.start,t.end))),list.map(t=>t.text))
  const choices=new Map([['BASE',['BASE_X2','不存在的字段','BASE_X1']]])
  assert.deepEqual(resolveFieldTerms(list,index,choices).matched,['BASE_X2','BASE_X1','a,b','有 空格','MOB6'])
  choices.set('BASE',[])
  assert.deepEqual(resolveFieldTerms(list,index,choices).matched,['a,b','有 空格','BASE_X1','MOB6'])
})
