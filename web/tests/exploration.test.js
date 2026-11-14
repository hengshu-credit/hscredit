import test from 'node:test'
import assert from 'node:assert/strict'
import {ref} from 'vue'
import {useExploration} from '../src/exploration.js'

function setup(submitCheck=()=>{}){
  const config=ref({dataset_id:'数据甲',dataset_revision:1,feature:'评分',other:null,bins:20,targets:[],positive_labels:{}})
  const requests=[],errors=[],jobs=new Map()
  const api=async(path,options)=>{
    requests.push({path,options})
    if(path==='/jobs'){
      const id=String(jobs.size+1),submitted=JSON.parse(options.body)
      await submitCheck(submitted)
      const job={id,config:submitted,result:{exploration:{summary:[{'特征名':'评分','平均值':50}],summary_columns:[{key:'特征名'}],distribution:[{'样本数':30}]}}}
      jobs.set(id,job);return job
    }
    if(path.endsWith('/stop'))return {status:'已停止'}
    return jobs.get(path.split('/').at(-1))
  }
  const state=useExploration({config,api,refresh:async()=>{},onError:message=>errors.push(message),clearError:()=>{}})
  const complete=()=>state.settle([{id:state.jobId.value,status:'成功'}])
  const submissions=()=>requests.filter(r=>r.path==='/jobs')
  return {config,requests,errors,jobs,state,complete,submissions}
}

test('重复进入页面和点击更新复用已完成结果，也不重复提交正在计算的任务',async()=>{
  const {state,complete,submissions,requests}=setup()
  await Promise.all([state.load(),state.load(),state.load()])
  assert.equal(submissions().length,1)
  await complete()
  const result=state.eda.value
  await state.load();await state.load()
  assert.equal(submissions().length,1)
  assert.equal(state.eda.value,result)
  assert.equal(state.summaryCurrent.value,true)
  assert.equal(requests.filter(r=>r.path.endsWith('/stop')).length,0)
})

test('恢复探索任务后仅重新读取结果，修改字段仍可运行新任务',async()=>{
  const {state,config,submissions}=setup()
  await state.load()
  const row={id:state.jobId.value,status:'成功'}, key=state.trackingKey.value
  state.reset()
  await state.view(row,key)
  assert.ok(state.eda.value)
  await state.load()
  assert.equal(submissions().length,1)
  config.value.feature='年龄'
  await state.load()
  assert.equal(submissions().length,2)
})

test('字典的首项和比较值参与概况缓存，服务端补齐空值仍能复用',async()=>{
  const {state,config,complete,jobs}=setup()
  Object.assign(config.value,{target:{甲:{field:'A'},乙:{field:'B',operator:'>',value:3}},target_order:['甲','乙']})
  await state.load()
  Object.assign(jobs.get(state.jobId.value).config.target.甲,{operator:null,value:null})
  await complete()
  assert.equal(state.summaryCurrent.value,true)
  config.value.target_order.reverse()
  assert.equal(state.summaryCurrent.value,false)
  await state.load();await complete()
  config.value.target.乙.value=7
  assert.equal(state.summaryCurrent.value,false)
})

test('切换 PSI 对照样本或其版本需要重新计算字段概况',async()=>{
  const {state,config,complete}=setup()
  await state.load();await complete()
  for (const ref of [{id:'验证',revision:1},{id:'验证',revision:2},{id:'OOT',revision:2},null]) {
    config.value.psi_dataset=ref
    assert.equal(state.summaryCurrent.value,false)
    await state.load();await complete()
    assert.equal(state.summaryCurrent.value,true)
  }
})

test('图表参数变化保留字段概况及其有效性',async()=>{
  const {state,config,complete,submissions}=setup()
  await state.load();await complete()
  const summary=state.eda.value.summary
  const columns=state.eda.value.summary_columns
  Object.assign(config.value,{feature:'收入',other:'评分',bins:5})
  assert.equal(state.summaryCurrent.value,true)
  await state.load()
  assert.equal(state.summaryCurrent.value,true)
  assert.equal(state.loading.value,true)
  assert.equal(state.summaryLoading.value,false)
  assert.equal(state.eda.value.summary,summary)
  assert.equal(submissions().length,2)
  await complete()
  assert.equal(state.eda.value.summary,summary)
  assert.equal(state.eda.value.summary_columns,columns)
  await state.load()
  assert.equal(submissions().length,2)
})

test('只有匹配当前标签的概况请求才显示计算状态',async()=>{
  const {state,config,complete}=setup()
  await state.load()
  assert.equal(state.summaryLoading.value,true)
  await complete()
  const previous=state.eda.value.summary
  Object.assign(config.value,{targets:['标签'],positive_labels:{标签:1}})
  await state.load()
  assert.equal(state.summaryLoading.value,true)
  config.value.positive_labels={标签:0}
  assert.equal(state.summaryLoading.value,false)
  await complete()
  assert.equal(state.summaryCurrent.value,false)
  assert.notEqual(state.eda.value.summary,previous)
})

test('标签增删、多标签及坏样本取值变动使旧摘要失效',async()=>{
  const {state,config,complete,submissions}=setup()
  await state.load();await complete()
  for(const [targets,positive_labels] of [[['标签甲'],{标签甲:1}],[['标签甲'],{标签甲:0}],[['标签甲','标签乙'],{标签甲:0,标签乙:'坏'}],[[],{}]]){
    Object.assign(config.value,{targets,positive_labels})
    assert.equal(state.summaryCurrent.value,false)
    await state.load();await complete()
    assert.equal(state.summaryCurrent.value,true)
  }
  assert.equal(submissions().length,5)
})

test('数据版本与数据集变化不展示旧摘要，重命名不影响结果',async()=>{
  const {state,config,complete,submissions}=setup()
  await state.load();await complete()
  config.value.name='改名'
  await state.load()
  assert.equal(submissions().length,1)
  for(const change of [{dataset_revision:2},{dataset_id:'数据乙'}]){
    Object.assign(config.value,change)
    assert.equal(state.summaryCurrent.value,false)
    state.reset()
    assert.equal(state.eda.value,null)
    await state.load();await complete()
    assert.equal(state.summaryCurrent.value,true)
  }
  assert.equal(submissions().length,3)
})

test('失败或停止后允许重试，错误结果不被缓存',async()=>{
  const {state,complete,submissions,errors}=setup()
  await state.load()
  await state.settle([{id:state.jobId.value,status:'失败',message:'计算失败'}])
  assert.deepEqual(errors,['计算失败'])
  assert.equal(state.summaryCurrent.value,false)
  await state.load()
  await state.settle([{id:state.jobId.value,status:'已停止'}])
  await state.load();await complete()
  assert.equal(submissions().length,3)
  assert.equal(state.summaryCurrent.value,true)
})

test('数据切换时迟到的旧任务结果不能覆盖当前结果或加载状态',async()=>{
  const config=ref({dataset_id:'数据甲',dataset_revision:1}),requests=[]
  let resolveResult
  const state=useExploration({config,refresh:async()=>{},clearError:()=>{},onError:assert.fail,api:async(path,options)=>{
    requests.push(path)
    if(path==='/jobs')return {id:config.value.dataset_id}
    if(path.endsWith('/stop'))return {}
    return new Promise(resolve=>{resolveResult=resolve})
  }})
  await state.load()
  const pending=state.settle([{id:'数据甲',status:'成功'}])
  config.value={dataset_id:'数据乙',dataset_revision:1};state.reset();await state.load()
  resolveResult({config:{dataset_id:'数据甲'},result:{exploration:{summary:['旧结果']}}})
  await pending
  assert.equal(state.eda.value,null)
  assert.equal(state.jobId.value,'数据乙')
  assert.equal(state.loading.value,true)
  assert.ok(!requests.some(path=>path.endsWith('/stop')))
})

test('改参只提交新任务，两次配置均保留，切换数据也不会停止已提交任务',async()=>{
  const {state,config,requests,jobs}=setup()
  await state.load()
  const first=state.jobId.value
  assert.equal(state.runningCurrentConfig.value,true)
  config.value.bins=8
  assert.equal(state.runningCurrentConfig.value,false)
  await state.load()
  const second=state.jobId.value
  assert.notEqual(first,second)
  assert.equal(jobs.get(first).config.bins,20)
  assert.equal(jobs.get(second).config.bins,8)
  await state.settle([{id:first,status:'成功'},{id:second,status:'排队中'}])
  assert.equal(state.currentJob.value.id,second)
  assert.equal(state.loading.value,true)
  state.reset()
  assert.ok(!requests.some(r=>r.path.endsWith('/stop')))
})

test('新配置未通过校验时原任务仍可完成并展示',async()=>{
  const {state,config,errors,complete}=setup(cfg=>{
    if(cfg.bins===0)throw Error('分组数无效')
  })
  await state.load()
  const first=state.jobId.value
  config.value.bins=0
  await state.load()
  assert.deepEqual(errors,['分组数无效'])
  assert.equal(state.jobId.value,first)
  assert.equal(state.loading.value,true)
  await complete()
  assert.equal(state.loading.value,false)
  assert.equal(state.eda.value.distribution[0]['样本数'],30)
  assert.equal(state.chartsCurrent.value,false)
})

test('已入队任务的提交响应迟到时只跟踪最新提交，两项任务都继续执行',async()=>{
  const config=ref({dataset_id:'甲', bins:20}), calls=[]
  let acceptFirst
  const state=useExploration({config,refresh:async()=>{},clearError:()=>{},onError:assert.fail,api:async(path,options)=>{
    calls.push(path)
    if(path!=='/jobs')return {}
    const cfg=JSON.parse(options.body)
    if(cfg.bins===20)await new Promise(resolve=>{acceptFirst=resolve})
    return {id:String(cfg.bins),config:cfg}
  }})
  const first=state.load()
  config.value.bins=8
  await state.load()
  acceptFirst()
  await first
  assert.deepEqual(calls,['/jobs','/jobs'])
  assert.equal(state.jobId.value,'8')
})
