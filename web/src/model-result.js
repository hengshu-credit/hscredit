import {shallowRef,watch} from 'vue'
import {datasetDraftKey,readDraft,writeDraft} from './form-drafts.js'

// 请求入队期间离开再返回，也能接到该样本、该页签最新一次提交的任务编号。
const selections=new Map()
export function modelResultSelection(dataset,compare){
  const key=datasetDraftKey(dataset,compare?'compare-result':'experiment-result')
  if(!selections.has(key)){
    const selected=shallowRef(readDraft(key) || {})
    let generation=0
    selections.set(key,{
      selected,
      begin(){return ++generation},
      select(token,jobId,explainId=''){
        if(token!==generation)return false
        selected.value={jobId,explainId}
        writeDraft(key,selected.value)
        return true
      },
    })
  }
  return selections.get(key)
}

export function followModelResult(id,job,api,onError){
  watch(id,async (value,_,cleanup)=>{
    if(!value){job.value=null;return}
    if(job.value?.id===value)return
    job.value=null
    const controller=new AbortController()
    cleanup(()=>controller.abort())
    try{
      const record=await api('/jobs/'+value,{signal:controller.signal})
      if(!controller.signal.aborted)job.value=record
    }catch(e){if(!controller.signal.aborted)onError(e.message)}
  },{immediate:true})
}
