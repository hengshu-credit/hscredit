<script setup>
import {computed, ref, watch} from 'vue'
import {Plus} from '@element-plus/icons-vue'
import {fieldValue, valueLabel} from '../table-filters'
import SelectedFieldNames from './SelectedFieldNames.vue'

const props = defineProps({field:Object, values:{type:Array, required:true}, rows:Array, loadValues:Function})
const emit = defineEmits(['change'])
const open = ref(false), query = ref(''), remoteValues = ref([]), loading = ref(false), error = ref(''), retry = ref(0)
const selectedLabels = computed(() => props.values.map(valueLabel))
const encodedValues = computed(() => props.values.map(value => JSON.stringify(value)))
const localValues = computed(() => !props.loadValues && props.field && open.value
  ? [...new Set(props.rows.map(row => fieldValue(row, props.field)))] : [])
const options = computed(() => {
  if (props.loadValues) return remoteValues.value
  const needle = query.value.toLowerCase()
  const matches = localValues.value.filter(value => valueLabel(value).toLowerCase().includes(needle))
  const exact = matches.find(value => valueLabel(value) === query.value)
  return [...new Set([...(exact === undefined ? [] : [exact]), ...matches.slice(0, 80)])]
})
const choices = computed(() => [...new Set([...options.value, ...props.values])])
const canCreate = computed(() => query.value !== '' && !loading.value &&
  !choices.value.some(value => value === query.value || valueLabel(value) === query.value))
watch([open, query, retry], async ([visible, keyword], _, onCleanup) => {
  remoteValues.value = []
  error.value = ''
  loading.value = false
  if (!visible || !props.field || !props.loadValues) return
  const controller = new AbortController()
  onCleanup(() => controller.abort())
  loading.value = true
  try {
    const values = await props.loadValues(props.field.key, keyword, controller.signal)
    if (!controller.signal.aborted) remoteValues.value = values
  } catch (e) {
    if (!controller.signal.aborted) error.value = e.message
  } finally {
    if (!controller.signal.aborted) loading.value = false
  }
})
function setOpen(visible) {
  open.value = visible
  if (!visible) query.value = ''
}
</script>

<template>
  <div class="filter-values">
    <el-select :model-value="encodedValues" multiple filterable remote remote-show-suffix
      default-first-option class="scrollable-tags" :reserve-keyword="false" :disabled="!field" :loading="loading"
      :remote-method="text => query = text" placeholder="选择或输入值" aria-label="添加筛选值" no-data-text="没有匹配的值"
      @update:model-value="encoded => emit('change', encoded.map(value => JSON.parse(value)))" @visible-change="setOpen">
      <template #prefix><span class="value-action"><el-icon><Plus/></el-icon>添加筛选值</span></template>
      <el-option v-if="canCreate" :value="JSON.stringify(query)" :label="`添加「${query}」`"/>
      <el-option v-for="value in choices" :key="JSON.stringify(value)" :label="valueLabel(value)" :value="JSON.stringify(value)"/>
      <template v-if="values.length>20" #tag><SelectedFieldNames :names="selectedLabels"/></template>
      <template #empty><p class="el-select-dropdown__empty">{{loading ? '正在读取筛选值…' : '没有匹配的值'}}</p></template>
      <template v-if="options.length >= 80" #footer><small class="muted">可输入更完整的值缩小范围</small></template>
    </el-select>
    <div v-if="error" role="alert">{{error}} <el-button link @click="retry++">重试</el-button></div>
  </div>
</template>

<style scoped>
.filter-values{min-width:0}
.value-action{display:inline-flex;align-items:center;gap:4px;color:var(--accent-text);font-size:12px;white-space:nowrap}
.filter-values :deep(.is-disabled) .value-action{color:var(--muted)}
.filter-values [role=alert]{color:var(--danger);margin-top:5px}
</style>
