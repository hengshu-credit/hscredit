import test from 'node:test'
import assert from 'node:assert/strict'
import {parameterRows, parseRows} from '../src/parameter-overrides.js'
import {orderParameters} from '../src/workflow-parameters.js'

const groups={binning:{params:{monotonic:{},missing_separate:{},user_splits:{},special_codes:{},cat_cutoff:{}},methods:{tree:{max_depth:{}}}}}

test('模型首屏优先当前模型的核心参数，同名报告参数保留在完整目录',()=>{
  const entries=[{id:'method',group:'评估报告'},{id:'cv'},{id:'model:C',group:'模型超参数'},{id:'features'},
    {id:'model:method',group:'模型超参数'},{id:'model_bins.method',scope:'model_bins'}]
  const ordered=orderParameters(entries,'model')
  assert.deepEqual(ordered.filter(s=>s.common).map(s=>s.id),['features','model:C','model:method','cv'])
  assert.deepEqual(ordered.filter(s=>!s.common).map(s=>s.id),['method','model_bins.method'])
})

test('树挖掘和二维分箱按各自的运行参数排序',()=>{
  const entries=['group','method','tree_depth','pair_binning.method_x'].map(id=>({id}))
  assert.deepEqual(orderParameters(entries,'binary').filter(s=>s.common).map(s=>s.id),['tree_depth','method'])
  assert.deepEqual(orderParameters(entries,'pair').filter(s=>s.common).map(s=>s.id),['pair_binning.method_x','method'])
})

test('参数按作用域和原始类型往返，删除后不再传值',()=>{
  const values={binning:{monotonic:'auto',missing_separate:false,user_splits:{x:[0,2]},special_codes:[],cat_cutoff:null,max_depth:2}}
  const rows=parameterRows(values)
  assert.deepEqual(JSON.parse(JSON.stringify(parseRows(rows,groups,'tree'))),values)
  assert.deepEqual(Object.keys(parseRows([],groups,'tree')),[])
})

test('切换算法、重复参数及无效 JSON 都不能静默提交',()=>{
  const rows=parameterRows({binning:{max_depth:2}})
  assert.throws(()=>parseRows(rows,groups,'quantile'),/max_depth/)
  assert.throws(()=>parseRows([...rows,...rows],groups,'tree'),/重复/)
  assert.throws(()=>parseRows([{...rows[0],text:'[1,'}],groups,'tree'),/JSON/)
})
