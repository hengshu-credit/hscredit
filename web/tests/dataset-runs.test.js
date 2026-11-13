import test from 'node:test'
import assert from 'node:assert/strict'
import {ref} from 'vue'
import {submitDatasetRuns} from '../src/dataset-runs.js'
import {useExploration} from '../src/exploration.js'

test('批量运行绑定各自版本，保留独立测试集和评估集的角色', async () => {
  const assets=[{id:'原表',revision:3},{id:'子集',revision:2}]
  const payload={kind:'model',test_dataset:{id:'外测'},evaluation_datasets:[{id:'评估'}]}
  const api=async (path, request)=>{
    assert.equal(path,'/jobs/batch')
    const {config,datasets}=JSON.parse(request.body)
    assert.deepEqual(config,{...payload,dataset_id:'原表',dataset_revision:3})
    assert.deepEqual(datasets,assets)
    return {jobs:datasets.map(d=>({dataset_id:d.id}))}
  }
  assert.deepEqual((await submitDatasetRuns(api,payload,assets)).map(j=>j.dataset_id),['原表','子集'])
  assert.equal(payload.dataset_id,undefined)
})

test('切换批内结果按各自的样本显示，调整所选样本后重新提交', async () => {
  const runDatasets=ref([{id:'源',revision:1},{id:'测试',revision:2}])
  const config=ref({dataset_id:'源',dataset_revision:1,feature:'评分',bins:20})
  const jobs=new Map(),submissions=[]
  const api=async(path,request)=>{
    if(path==='/jobs/batch'){
      const payload=JSON.parse(request.body),batch=String(submissions.push(payload))
      const created=payload.datasets.map(d=>{
        const j={id:batch+d.id,batch_id:batch,status:'成功',dataset_id:d.id,dataset_revision:d.revision,
          config:{...payload.config,dataset_id:d.id,dataset_revision:d.revision},result:{exploration:{sample:d.id}}}
        jobs.set(j.id,j);return j
      })
      return {jobs:created}
    }
    return jobs.get(path.split('/').at(-1))
  }
  const state=useExploration({config,runDatasets,api,refresh:async()=>{},clearError:()=>{},onError:assert.fail})
  await state.load();await state.settle([...jobs.values()])
  assert.equal(state.eda.value.sample,'源')
  await state.view(jobs.get('1测试'))
  assert.equal(state.eda.value.sample,'测试')
  assert.equal(state.summaryCurrent.value,true)
  await state.load()
  assert.equal(submissions.length,1)
  runDatasets.value.push({id:'验证',revision:1})
  await state.load()
  assert.equal(submissions.length,2)
  assert.equal(submissions[1].datasets.length,3)
})
