<script setup>
import {ref,watch} from 'vue'
import {api} from '../api'
import DataTable from './DataTable.vue'

const props=defineProps({dataset:{type:Object,required:true}})
const result=ref(null),error=ref(''),retry=ref(0)
watch([()=>props.dataset.id,()=>props.dataset.revision||1,retry],async ([id,revision],_,onCleanup)=>{
  const controller=new AbortController()
  onCleanup(()=>controller.abort())
  result.value=null
  error.value=''
  try{
    const summary=await api(`/datasets/${id}/summary?revision=${revision}`,{signal:controller.signal})
    if(!controller.signal.aborted)result.value=summary
  }catch(e){
    if(!controller.signal.aborted)error.value=e.message
  }
},{immediate:true})
</script>

<template>
  <section class="panel" aria-label="字段概况" :aria-busy="!result&&!error">
    <div class="panel-heading"><h2>字段概况</h2><span class="muted">{{dataset.name}} · 无标签</span></div>
    <DataTable v-if="result" field-filters :rows="result.summary" :column-groups="result.summary_columns"/>
    <div v-else-if="error" role="alert">{{error}} <el-button link type="primary" @click="retry++">重试</el-button></div>
    <p v-else class="muted" role="status">正在读取字段概况…</p>
  </section>
</template>
