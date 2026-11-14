import test from 'node:test'
import assert from 'node:assert/strict'
import {effectScope,nextTick,ref} from 'vue'
import {modelResultSelection,followModelResult} from '../src/model-result.js'

test('切换页签后继续跟踪延迟提交的任务，旧请求和其他版本不会覆盖当前报告',async()=>{
  const dataset={id:'验证',revision:1}
  const first=modelResultSelection(dataset,false)
  const slow=first.begin()
  const reopened=modelResultSelection(dataset,false)
  const displayed=ref(null), errors=[], calls=[]
  const scope=effectScope()
  scope.run(()=>followModelResult(()=>reopened.selected.value.jobId,displayed,async path=>{
    calls.push(path)
    return {id:path.slice(6),status:'成功'}
  },message=>errors.push(message)))
  assert.equal(first.select(slow,'第一项'),true)
  await nextTick()
  await Promise.resolve()
  assert.equal(displayed.value.id,'第一项')
  const current=reopened.begin()
  assert.equal(reopened.select(current,'第二项'),true)
  assert.equal(first.select(slow,'过时任务'),false)
  await nextTick()
  await Promise.resolve()
  assert.equal(displayed.value.id,'第二项')
  assert.deepEqual(calls,['/jobs/第一项','/jobs/第二项'])
  assert.deepEqual(errors,[])
  assert.deepEqual(modelResultSelection(dataset,true).selected.value,{})
  assert.deepEqual(modelResultSelection({...dataset,revision:2},false).selected.value,{})
  scope.stop()
})
