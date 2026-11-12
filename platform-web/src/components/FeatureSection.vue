<script setup>
import { computed, ref, watch } from 'vue'
import DataTable from './DataTable.vue'
import ReportImage from './ReportImage.vue'
const props=defineProps({job:String,feature:String,dataset:String,tables:Array,images:Array})
const prefix=computed(()=>props.dataset+' / '+props.feature+' / ')
const tables=computed(()=>props.tables.filter(([name])=>name.startsWith(prefix.value)))
const plots=computed(()=>props.images.filter(img=>img.title.startsWith(prefix.value)))
const table=ref(''),plot=ref('')
watch(tables,val=>{table.value=val[0]?.[0]||''},{immediate:true})
watch(plots,val=>{plot.value=val[0]?.file||''},{immediate:true})
const rows=computed(()=>tables.value.find(([key])=>key===table.value)?.[1]||[])
const image=computed(()=>plots.value.find(img=>img.file===plot.value))
</script>
<template><section class="variable-section"><h3>{{feature}}</h3>
  <el-select v-if="tables.length>1" v-model="table" aria-label="选择分箱表"><el-option v-for="([name]) in tables" :key="name" :value="name" :label="name.slice(prefix.length)"/></el-select>
  <DataTable :rows="rows"/>
  <el-select v-if="plots.length>1" v-model="plot" filterable aria-label="选择变量图表"><el-option v-for="img in plots" :key="img.file" :value="img.file" :label="img.title.slice(prefix.length)"/></el-select>
  <ReportImage v-if="image" :job="job" :file="image.file" :title="image.title.slice(prefix.length)"/>
</section></template>
<style scoped>.variable-section{min-width:0;border-top:1px solid var(--border);padding:10px 0}.variable-section :deep(.el-select){max-width:360px}</style>
