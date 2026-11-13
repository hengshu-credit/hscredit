import test from 'node:test'
import assert from 'node:assert/strict'
import {placeField} from '../src/explore-fields.js'

const fields=[
  {name:'金额',kind:'numeric',unique:100,missing:0},
  {name:'期限',kind:'numeric',unique:6,missing:0},
  {name:'渠道',kind:'category',unique:5,missing:1},
  {name:'日期',kind:'date',unique:30,missing:1},
]

test('单变量散点只接受两个不同的数值字段，坐标轴可互换',()=>{
  const axes=['金额','期限']
  assert.deepEqual(placeField('single',axes,'',0,'期限',fields),{axes:['期限','金额'],group:''})
  assert.ok(placeField('single',axes,'',1,'渠道',fields).error)
  assert.deepEqual(axes,['金额','期限'])
  assert.deepEqual(placeField('single',['金额'],'',0,'日期',fields),{axes:['日期'],group:''})
})

test('交互布局保持坐标顺序并可将变量与分组标签互换',()=>{
  assert.ok(placeField('multi',[],'',2,'日期',fields).error)
  assert.deepEqual(placeField('multi',['金额'],'',1,'日期',fields),{axes:['金额','日期'],group:''})
  assert.deepEqual(placeField('multi',['金额','期限'],'渠道','group','期限',fields),{axes:['金额','渠道'],group:'期限'})
  assert.deepEqual(placeField('multi',['金额','期限'],'渠道',1,'渠道',fields),{axes:['金额','渠道'],group:'期限'})
  assert.deepEqual(placeField('multi',['金额','期限'],'渠道',2,'金额',fields),{axes:['期限','金额'],group:'渠道'})
  assert.ok(placeField('multi',['金额'],'','group','日期',fields).error)
})
