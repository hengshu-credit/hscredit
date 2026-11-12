import test from 'node:test'
import assert from 'node:assert/strict'
import {parameterRows, parseRows} from '../src/parameter-overrides.js'

const groups={binning:{params:{monotonic:{},missing_separate:{},user_splits:{},special_codes:{},cat_cutoff:{}},methods:{tree:{max_depth:{}}}}}

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
