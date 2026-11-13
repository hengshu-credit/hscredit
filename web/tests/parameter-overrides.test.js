import test from 'node:test'
import assert from 'node:assert/strict'
import {parameterRows, parseRows, parameterDocument, readParameterDocument} from '../src/parameter-overrides.js'
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

const specs = [
  {key:'method', default:'quantile', choices:['quantile','tree'].map(value=>({value}))},
  {key:'max_n_bins', default:5, min:2, max:50},
]
const plain = obj => JSON.parse(JSON.stringify(obj))

test('表单直接输入文本和布尔值，JSON 改名、改值、删除后可继续用表单编辑',()=>{
  const rows = [
    {scope:'binning', name:'monotonic', text:'auto_asc_desc'},
    {scope:'binning', name:'missing_separate', text:'true'},
  ]
  const document = parameterDocument(specs, {method:'tree',max_n_bins:8}, parseRows(rows,groups,'tree'), groups)
  assert.deepEqual(document, {method:'tree',max_n_bins:8,monotonic:'auto_asc_desc',missing_separate:true})
  delete document.monotonic
  delete document.max_n_bins
  Object.assign(document, {missing_separate:false, max_depth:2, special_codes:[0], cat_cutoff:null})
  const {values,overrides} = readParameterDocument(JSON.stringify(document),specs,groups)
  assert.deepEqual(values, {method:'tree',max_n_bins:5})
  assert.deepEqual(plain(overrides), {binning:{missing_separate:false,max_depth:2,special_codes:[0],cat_cutoff:null}})
  assert.deepEqual(plain(parseRows(parameterRows(overrides),groups,values.method)),plain(overrides))
})

test('JSON 字符串中的保留字、结构和空白往返后保留原始类型',()=>{
  for(const value of ['auto', 'true', 'False', 'null', '12', '[]', '{', '', '  名称 ', '"标签"', false, 0, [], {x:[1,null]}]){
    const {overrides} = readParameterDocument(JSON.stringify({monotonic:value}),[],groups)
    const rows = parameterRows(overrides)
    assert.deepEqual(parseRows(rows,groups).binning.monotonic,value)
  }
})

test('模型参数、搜索空间对象和多个用途的同名值独立往返',()=>{
  const modelSpecs = [...specs, {key:'model:method',apiName:'method',default:'platt'},
    {key:'model:max_depth',apiName:'max_depth',default:3,jsonValue:true}, {key:'search_spaces',default:'{}',type:'json'}]
  const modelGroups = {roc:{params:{title:{}}}, ks:{params:{title:{}}}}
  const doc = {method:'tree','model.method':'beta',max_depth:null,search_spaces:{rf:{max_depth:[2,5]}},'roc.title':'ROC','ks.title':'KS'}
  const {values,overrides} = readParameterDocument(JSON.stringify(doc),modelSpecs,modelGroups)
  assert.equal(values['model:method'],'beta')
  assert.equal(values['model:max_depth'],null)
  assert.deepEqual(JSON.parse(values.search_spaces),doc.search_spaces)
  assert.deepEqual(parameterDocument(modelSpecs,values,overrides,modelGroups),{...doc,max_n_bins:5})
  assert.throws(()=>readParameterDocument('{"title":"?"}',modelSpecs,modelGroups),/roc.title.*ks.title/)
})

test('无效文档、重复键、错误类型和不适用于当前方法的参数被拒绝',()=>{
  for (const json of ['{', '[]', 'null','{"missing_separate":true,"missing_separate":false}',
    '{"monotonic":true,"binning.monotonic":false}', '{"missing_sep":true}', '{"max_depth":4}',
    '{"max_n_bins":"6"}', '{"max_n_bins":80}', '{"method":"unknown"}', '{"toString.title":42}']){
    assert.throws(()=>readParameterDocument(json,specs,groups))
  }
})
