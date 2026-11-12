import test from 'node:test'
import assert from 'node:assert/strict'
import {chosenFormat,crossCounts,distributionRows,fieldChart,fieldChoices,pairChart,pairChoices} from '../src/explore-formats.js'

test('字段图形独立按类型选择，分布表和累计曲线采用相同全量频数',()=>{
  const dist={feature:'金额',kind:'numeric',distribution:[{区间:'低',样本数:1},{区间:'高',样本数:3}],five_number:[1,2,4,6,20]}
  assert.equal(chosenFormat(fieldChoices('category'),'box'),'rank')
  assert.equal(chosenFormat(fieldChoices('boolean'),undefined),'pie')
  assert.deepEqual(distributionRows(dist).map(r=>r['占比（%）']),[25,75])
  assert.deepEqual(fieldChart(dist,'cumulative').series[0].data,[25,100])
  assert.deepEqual(fieldChart(dist,'box').series[0].data,[[1,2,4,6,20]])
})

test('交叉图按类型推荐，频数表和热图按所选分组合并同一格',()=>{
  assert.equal(pairChoices(['date','numeric'])[0][0],'means')
  assert.equal(pairChoices(['numeric','numeric'])[0][0],'scatter')
  assert.equal(pairChoices(['category','boolean'])[0][0],'stack')
  const axes=[{name:'渠道',categories:['门店']},{name:'状态',categories:['是']}]
  const pair={indices:[0,1],order:[['门店'],['是']],density:[{x:'门店',y:'是',标签:'甲',样本数:2},{x:'门店',y:'是',标签:'乙',样本数:3}]}
  assert.deepEqual(crossCounts(pair,null),[{x:'门店',y:'是',count:5}])
  assert.deepEqual(pairChart(pair,axes,'density','乙').series[0].data,[[0,0,3]])
  assert.deepEqual(pairChart(pair,axes,'stack').series[0].data,[5])
})
