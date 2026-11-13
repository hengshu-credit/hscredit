import test from 'node:test'
import assert from 'node:assert/strict'
import {profileChart,profileValue} from '../src/column-profile.js'

test('四种字段分布保留后端计数并采用对应图形',()=>{
  const distribution=[{label:'甲',count:70},{label:'乙',count:9}]
  for(const [kind,type] of [['numeric','bar'],['category','bar'],['date','line'],['boolean','pie']]){
    const chart=profileChart({kind,distribution})
    assert.equal(chart.series[0].type,type)
    assert.deepEqual(kind==='boolean'?chart.series[0].data.map(p=>p.value):chart.series[0].data,[70,9])
    if(kind==='category')assert.equal(chart.yAxis.type,'category')
    if(kind==='numeric')assert.equal(chart.series[0].barCategoryGap,'0%')
  }
})
test('缺失、百分比、布尔及大整数文本准确展示',()=>{
  assert.equal(profileValue('平均值',null),'—')
  assert.equal(profileValue('缺失率',0.125),'12.50%')
  assert.equal(profileValue('众数',false),'假')
  assert.equal(profileValue('众数','19859456400262761'),'19859456400262761')
})
