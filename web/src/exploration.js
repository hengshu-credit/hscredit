// 探索结果独立于页面生命周期；字段概况只随数据版本、标签及坏样本取值失效。
import {computed,ref,shallowRef} from 'vue'
import {submitDatasetRuns} from './dataset-runs.js'

export const runningStatuses=['排队中','运行中','停止中']

function summaryKey(config){
  const psi=config.psi_dataset?[config.psi_dataset.id,config.psi_dataset.revision]:null
  if (config.target && typeof config.target==='object') {
    const definitions=(config.target_order||Object.keys(config.target)).map(name=>{
      const rule=config.target[name]
      return [name,Array.isArray(rule)?rule:[rule.field,rule.operator??null,rule.value??null]]
    })
    return JSON.stringify([config.dataset_id,config.dataset_revision||1,config.library_params?.summary||{},psi,'dict',definitions,!!config.del_grey,config.handle_missing_target||'drop'])
  }
  const labels=config.label_mode==='overdue'
    ? [config.overdue,config.dpds,config.overdue_operator,config.del_grey]
    : [config.targets||[],Object.entries(config.positive_labels||{}).sort(([a],[b])=>a.localeCompare(b))]
  return JSON.stringify([config.dataset_id,config.dataset_revision||1,config.library_params?.summary||{},psi,config.label_mode||'target',labels,config.handle_missing_target||'drop'])
}
function explorationKey(config){
  return JSON.stringify([summaryKey(config),config.feature||null,config.other||null,config.bins??20,config.freq||'M',config.workflow?config.features:null,config.group||null])
}

export function useExploration({config,api,refresh,onError,clearError,runDatasets}){
  const eda=ref(null),jobId=ref(''),applied=ref(null)
  const currentJob=ref(null)
  const pending = shallowRef(null), following = ref(false)
  const loading = computed(() => !!pending.value || following.value)
  const sources=computed(()=>runDatasets?.value??[{id:config.value.dataset_id,revision:config.value.dataset_revision||1}])
  const bindings=computed(()=>sources.value.map(d=>({id:d.id,revision:d.revision||1})))
  const runKey=computed(()=>JSON.stringify([explorationKey(config.value),bindings.value]))
  const requested=ref('')
  const runningCurrentConfig=computed(()=>pending.value?.key===runKey.value || (following.value && requested.value===runKey.value))
  const displayConfig=computed(()=>{
    const source=sources.value.find(d=>d.id===applied.value?.dataset_id)
    return source?{...config.value,dataset_id:source.id,dataset_revision:source.revision||1}:config.value
  })
  const summaryCurrent=computed(()=>!!eda.value&&!!applied.value&&summaryKey(applied.value)===summaryKey(displayConfig.value))
  const summaryLoading = computed(() => loading.value && !summaryCurrent.value
    && (pending.value?.summary ?? (currentJob.value?.config && summaryKey(currentJob.value.config))) === summaryKey(displayConfig.value))
  const chartsCurrent=computed(()=>!!eda.value&&!!applied.value&&explorationKey(applied.value)===explorationKey(displayConfig.value))
  let revision=0,settled='',lastRun=''

  function reset(){
    revision++
    pending.value=null;following.value=false
    jobId.value='';currentJob.value=null;eda.value=null;applied.value=null;requested.value='';settled='';lastRun=''
  }

  async function load(){
    const snapshot=JSON.parse(JSON.stringify(config.value))
    if(!snapshot.dataset_id)return
    const key=runKey.value
    if(pending.value?.key===key)return
    if((following.value&&requested.value===key)||(!following.value&&eda.value&&lastRun===key)){
      pending.value=null
      return
    }
    const submission={key,summary:summaryKey(snapshot)}
    pending.value=submission
    clearError()
    try{
      const [job]=await submitDatasetRuns(api,{kind:'explore',...snapshot},bindings.value)
      if(pending.value!==submission)return
      // 新任务入队后才切换当前结果。其他已入队任务继续执行。
      revision++;requested.value=key;settled='';lastRun='';following.value=true
      currentJob.value=job;jobId.value=job.id;await refresh()
    }catch(e){if(pending.value===submission)onError(e.message)}
    finally{if(pending.value===submission)pending.value=null}
  }

  async function settle(jobs){
    const id=jobId.value,row=jobs.find(j=>j.id===id),rev=revision
    if(!row||runningStatuses.includes(row.status)||settled===id)return
    settled=id
    try{
      if(row.status==='成功'){
        const job=await api(`/jobs/${id}`)
        if(rev!==revision)return
        const hasTargets = !!(job.config.targets?.length || job.config.overdue?.length && job.config.label_mode==='overdue' || job.config.target && typeof job.config.target==='object')
        const result = {...job.result.exploration,interaction:job.result.interaction,job:{id:job.id,datasetId:job.dataset_id,revision:job.dataset_revision||1,name:job.dataset_name,hasTargets}}
        const completedConfig = {...job.config,dataset_revision:job.dataset_revision||job.config.dataset_revision||1}
        // 仅图表变化时保留概况行及表头的引用，避免 DataTable 重置筛选和分页。
        const reuseSummary = eda.value && applied.value && summaryKey(applied.value) === summaryKey(completedConfig)
        eda.value = reuseSummary
          ? {...result, summary: eda.value.summary, summary_columns: eda.value.summary_columns}
          : result
        currentJob.value=job;lastRun=requested.value
        applied.value=completedConfig
      }else if(row.status==='失败'){onError(row.message)}
    }catch(e){if(rev===revision){onError(e.message);settled=''}}
    finally{if(rev===revision)following.value=false}
  }

  async function view(row){
    revision++;settled='';pending.value=null;lastRun='';clearError()
    jobId.value=row.id;currentJob.value=row;eda.value=null;applied.value=null;following.value=true
    await settle([row])
  }
  return {eda,loading,following,runningCurrentConfig,jobId,currentJob,summaryCurrent,summaryLoading,chartsCurrent,load,settle,reset,view}
}
