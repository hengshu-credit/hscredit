<script setup>
import {computed, onBeforeUnmount, ref, watch} from 'vue'
import {api} from '../api'
import {profileChart, profileValue} from '../column-profile'
import Chart from './Chart.vue'

const props = defineProps({datasetId: String, revision: Number, column: String, visible: Boolean, themeKey: String})
const emit = defineEmits(['resize'])
const profile = ref(null), loading = ref(false), error = ref('')
const option = computed(() => profile.value && profileChart(profile.value))
const summary = computed(() => Object.entries(profile.value?.summary || {}).filter(([name]) => name !== '特征名'))
let controller

async function load() {
  if (!props.visible || loading.value || profile.value) return
  const request = new AbortController()
  controller = request
  loading.value = true
  error.value = ''
  try {
    const query = new URLSearchParams({column: props.column, revision: props.revision})
    const result = await api(`/datasets/${props.datasetId}/column-profile?${query}`, {signal: request.signal})
    if (!request.signal.aborted) profile.value = result
  } catch (e) {
    if (!request.signal.aborted) error.value = e.message
  } finally {
    if (!request.signal.aborted) {
      loading.value = false
      emit('resize')
    }
  }
}

watch(() => [props.datasetId, props.revision, props.column], () => {
  controller?.abort()
  profile.value = null
  loading.value = false
  load()
}, {immediate: true})
watch(() => props.visible, open => { if (open) load() })
onBeforeUnmount(() => controller?.abort())
</script>
<template>
  <slot name="header" :kind="profile?.kind"/>
  <div v-if="loading" class="profile-loading" role="status" aria-busy="true"><p class="muted">正在计算完整列的摘要与分布…</p><el-skeleton :rows="5" animated/></div>
  <div v-else-if="error" role="alert"><p>{{error}}</p><el-button size="small" @click="load">重新加载</el-button></div>
  <template v-else-if="profile">
    <p class="profile-scope">全量 {{profile.rows.toLocaleString()}} 行 · 版本 {{revision}} · 不受预览分页和搜索影响</p>
    <div class="profile-cards"><div v-for="name in ['缺失数','缺失率','唯一值数']" :key="name"><span>{{name}}</span><strong>{{profileValue(name,profile.summary[name])}}</strong></div></div>
    <h3>{{{numeric:'数值分布',category:'类别频数',date:'时间分布',boolean:'布尔占比'}[profile.kind]}}</h3>
    <Chart v-if="visible&&profile.distribution.some(p=>p.count>0)" :option="option" :theme-key="themeKey" class="profile-chart" :style="{height:profile.kind==='category'?Math.max(210,profile.distribution.length*22+45)+'px':'230px'}"/>
    <el-empty v-else-if="!profile.distribution.some(p=>p.count>0)" description="暂无可绘制的有效值" :image-size="45"/>
    <p class="profile-note">{{profile.note}}<span v-if="profile.invalid"> 已排除 {{profile.invalid.toLocaleString()}} 个无效值。</span></p>
    <h3>字段概况 <small>summary · 无标签</small></h3>
    <dl class="profile-summary" aria-label="完整字段摘要"><div v-for="[name,value] in summary" :key="name"><dt>{{name}}</dt><dd :title="profileValue(name,value)">{{profileValue(name,value)}}</dd></div></dl>
  </template>
</template>

<style>
.profile-scope,.profile-note{font-size:12px;line-height:1.6;color:var(--muted);margin:10px 0}
.column-profile .profile-scope{margin-top:4px}
.profile-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.profile-cards>div{background:var(--page);padding:10px}.profile-cards span{display:block;font-size:12px;color:var(--muted)}.profile-cards strong{display:block;margin-top:4px;font-size:18px;font-variant-numeric:tabular-nums}
.column-profile h3{font-size:13px;font-weight:600;margin:14px 0 8px}.column-profile h3 small{font-weight:400;color:var(--muted);margin-left:8px}.profile-chart.chart{min-height:0;width:100%}
.profile-summary{margin:0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 16px;font-size:12px}.profile-summary>div{display:flex;gap:8px;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);min-width:0}.profile-summary dt{color:var(--muted);flex-shrink:0}.profile-summary dd{margin:0;min-width:0;overflow-wrap:anywhere;text-align:right;font-variant-numeric:tabular-nums}.profile-loading{padding:16px 0}
@media(max-width:500px){.profile-summary{grid-template-columns:minmax(0,1fr)}}
</style>
