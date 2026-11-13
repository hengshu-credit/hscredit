<script setup>
import {ref,watch,onMounted,onBeforeUnmount} from 'vue'
import {apiBlob,artifactBlob} from '../api'
const props=defineProps({job:String,file:String,title:String,path:String,lazy:Boolean})
const host=ref(null),visible=ref(!props.lazy)
let observer
onMounted(()=>{
  if (!props.lazy || !window.IntersectionObserver) {visible.value=true;return}
  observer=new IntersectionObserver(entries=>{
    if(entries.some(entry=>entry.isIntersecting)){visible.value=true;observer.disconnect()}
  },{rootMargin:'240px'})
  observer.observe(host.value)
})
const src=ref(''),error=ref('');let revision=0
watch(()=>[props.job,props.file,props.path,visible.value],async()=>{
  if(!visible.value)return
  const rev=++revision
  if(src.value)URL.revokeObjectURL(src.value)
  src.value='';error.value=''
  try{
    const blob=await (props.path?apiBlob(props.path):artifactBlob(props.job,props.file))
    if(rev===revision)src.value=URL.createObjectURL(blob)
  }catch(e){if(rev===revision)error.value=e.message}
},{immediate:true})
onBeforeUnmount(()=>{revision++;observer?.disconnect();if(src.value)URL.revokeObjectURL(src.value)})
</script>
<template><div ref="host" class="report-image"><h3>{{title}}</h3><el-image v-if="src" :src="src" :preview-src-list="[src]" :alt="title" fit="contain"/><el-alert v-else-if="error" :title="error" type="error"/><el-skeleton v-else :rows="5" animated/></div></template>
<style scoped>
.report-image :deep(.el-image-viewer__btn){background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:var(--radius);opacity:1}
</style>
