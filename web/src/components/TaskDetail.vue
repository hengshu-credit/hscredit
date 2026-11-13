<script setup>
import {computed, ref, watch} from 'vue'
import {ElMessage} from 'element-plus'
import {api, download} from '../api'
import {runningStatuses} from '../exploration'
import {taskName, taskStatusType} from '../task-center'
import BatchResultPicker from './BatchResultPicker.vue'
import WorkbookReport from './WorkbookReport.vue'
import ReportImage from './ReportImage.vue'
import DataTable from './DataTable.vue'

const props = defineProps({id:String, jobs:Array, stopping:Array, embedded:Boolean})
const emit = defineEmits(['open','edit','stop','select'])
const summary = computed(() => props.jobs.find(job => job.id === props.id))
const detail = ref(null), error = ref(''), retry = ref(0), imageIndex = ref(0)
watch([() => props.id, () => summary.value?.status, retry], async (_, __, onCleanup) => {
  const controller = new AbortController()
  onCleanup(() => controller.abort())
  error.value = ''
  try {
    detail.value = await api('/jobs/' + props.id, {signal:controller.signal})
  } catch (e) { if (!controller.signal.aborted) error.value = e.message }
}, {immediate:true})
const job = computed(() => detail.value?.result ? detail.value : summary.value || detail.value)
const files = computed(() => job.value?.artifacts || [])
const images = computed(() => [...(job.value?.result?.images || []), ...Object.values(job.value?.result?.visuals || {}).flatMap(v => v.images || [])])
async function save(file) {
  try { await download(props.id, file) } catch (e) { ElMessage.error(e.message) }
}
</script>
<template>
  <div v-if="error" class="panel" role="alert">{{error}} <el-button @click="retry++">重新加载</el-button></div>
  <template v-if="job">
    <section class="panel">
      <div class="panel-heading"><div><h2>{{embedded?'本次任务结果':taskName(job)}} <el-tag :type="taskStatusType(job.status)">{{job.status}}</el-tag></h2>
        <p class="muted">{{job.dataset_name}} · 版本 {{job.dataset_revision||1}} · {{new Date(job.created_at).toLocaleString()}}</p></div>
        <div><el-button v-if="runningStatuses.includes(job.status)" :loading="job.status==='停止中'||stopping.includes(id)" @click="emit('stop',job)">停止任务</el-button>
          <el-button v-if="embedded" @click="emit('open',job)">任务详情</el-button>
          <el-button v-else @click="emit('edit',job)">重新配置并执行</el-button></div>
      </div>
      <BatchResultPicker :job="job" @select="emit(embedded?'select':'open',$event)"/>
      <el-progress v-if="runningStatuses.includes(job.status)" :percentage="job.progress"/>
      <p v-if="job.status!=='成功'||job.message!=='任务完成'" role="status">{{job.message}}</p>
      <template v-if="job.status==='失败'"><div v-for="(rows,name) in job.result?.tables" :key="name"><h3>{{name}}</h3><DataTable :rows="rows"/></div></template>
    </section>
    <WorkbookReport v-if="job.status==='成功'&&files.some(f=>f.endsWith('.xlsx'))" :key="id" :job="job"/>
    <section v-if="!embedded" class="panel"><el-collapse>
      <el-collapse-item v-if="images.length" title="报告图表" name="images">
        <el-select v-model="imageIndex" aria-label="选择报告图表"><el-option v-for="(img,i) in images" :key="img.file" :value="i" :label="img.title"/></el-select>
        <ReportImage v-if="images[imageIndex]" :job="id" :file="images[imageIndex].file" :title="images[imageIndex].title"/>
      </el-collapse-item>
      <el-collapse-item v-if="files.some(f=>!f.endsWith('.xlsx')&&!f.endsWith('.png'))" title="其他产物"><div class="download-bar">
        <el-button v-for="file in files.filter(f=>!f.endsWith('.xlsx')&&!f.endsWith('.png'))" :key="file" @click="save(file)">{{file}}</el-button></div></el-collapse-item>
      <el-collapse-item title="本次运行配置"><pre>{{JSON.stringify(job.config,null,2)}}</pre></el-collapse-item>
    </el-collapse></section>
  </template>
  <p v-else-if="!error" class="panel muted" role="status">正在读取任务详情…</p>
</template>
