<script setup>
import {computed, nextTick, ref} from 'vue'
import {Close, Filter, Plus, RefreshLeft} from '@element-plus/icons-vue'
import TableFilterValues from './TableFilterValues.vue'

const props = defineProps({
  modelValue: {type:Array, required:true},
  fields: {type:Array, required:true},
  rows: {type:Array, default:() => []},
  loadValues: Function,
  hasSearch: Boolean,
})
const emit = defineEmits(['update:modelValue', 'reset'])
const blankSelect = ref(null)
const selectedKeys = computed(() => new Set(props.modelValue.map(row => row.key)))
const byKey = computed(() => new Map(props.fields.map(field => [field.key, field])))
const canAdd = computed(() => !selectedKeys.value.has('') && props.fields.some(field => !selectedKeys.value.has(field.key)))
async function addField() {
  if (!canAdd.value) return
  emit('update:modelValue', [...props.modelValue, {key:'', values:[]}])
  await nextTick()
  blankSelect.value?.focus()
}
function updateRow(index, key, values) {
  emit('update:modelValue', props.modelValue.map((row, i) => i === index ? {key, values} : row))
}
function reset() {
  emit('update:modelValue', [])
  emit('reset')
}
</script>

<template>
  <div class="table-filters" role="group" aria-label="表格筛选">
    <div class="filter-toolbar">
      <span class="filter-title"><el-icon><Filter/></el-icon>筛选条件</span>
      <el-button :icon="Plus" plain type="primary" :disabled="!canAdd" @click="addField">添加筛选字段</el-button>
      <div v-if="$slots.search" class="filter-search"><slot name="search"/></div>
      <el-button class="filter-reset" link :icon="RefreshLeft" :disabled="!modelValue.length && !hasSearch" @click="reset">重置</el-button>
    </div>
    <div v-for="(row, index) in modelValue" :key="row.key" class="filter-row" role="group" :aria-label="byKey.get(row.key)?.label || '新筛选条件'">
      <el-select :ref="el => { if (!row.key) blankSelect = el }" :model-value="row.key" filterable automatic-dropdown
        placeholder="选择筛选字段" aria-label="筛选字段" @update:model-value="key => updateRow(index, key, [])">
        <el-option v-for="field in fields" :key="field.key" :value="field.key" :label="field.label"
          :disabled="field.key !== row.key && selectedKeys.has(field.key)"/>
      </el-select>
      <TableFilterValues :field="byKey.get(row.key)" :values="row.values" :rows="rows" :load-values="loadValues"
        @change="values => updateRow(index, row.key, values)"/>
      <el-button class="remove-field" text :icon="Close" title="移除筛选字段" aria-label="移除筛选字段"
        @click="emit('update:modelValue', modelValue.filter((_, i) => i !== index))"/>
    </div>
  </div>
</template>

<style scoped>
.table-filters{container-type:inline-size;margin:0 0 14px;padding:12px 14px;background:var(--soft);border:1px solid var(--line-soft);border-radius:var(--radius)}
.filter-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px 12px}
.filter-title{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600;white-space:nowrap}
.filter-title .el-icon{color:var(--accent-text)}
.filter-search{flex:1 1 150px;min-width:0;max-width:320px;margin-left:auto}
.filter-toolbar .filter-reset{margin-left:auto}
.filter-search+.filter-reset{margin-left:0}
.filter-row{display:grid;grid-template-columns:minmax(140px,1fr) minmax(160px,2.2fr) auto;gap:8px 12px;align-items:start;margin-top:8px}
.filter-toolbar+.filter-row{border-top:1px solid var(--line-soft);padding-top:12px;margin-top:12px}
.filter-row>*{min-width:0}
.remove-field{color:var(--muted)}
@container (max-width:540px){
  .filter-search{order:1;flex-basis:100%;max-width:none}
  .filter-toolbar .filter-reset{margin-left:auto}
  .filter-row{grid-template-columns:minmax(0,1fr) auto}
  .filter-row .filter-values{grid-column:1/-1;grid-row:2}
  .remove-field{grid-column:2;grid-row:1}
}
@container (max-width:340px){
  .filter-title{display:none}
}
</style>
