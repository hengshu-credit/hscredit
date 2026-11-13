<script setup>
import {computed, inject, ref, watch} from 'vue'
import {Search} from '@element-plus/icons-vue'
import {api} from '../api'
import {useTableFilters} from '../table-filters'
import ColumnProfile from './ColumnProfile.vue'
import TableFilters from './TableFilters.vue'

const props = defineProps({dataset: {type:Object, required:true}, themeKey:String})
const theme = inject('platformTheme')
const fields = computed(() => props.dataset.columns.map(({name}) => ({key:name, label:name})))
const {drafts, conditions} = useTableFilters()
const page = ref(1), search = ref(''), keyword = ref('')
const preview = ref({rows:[], total:0}), loading = ref(false), error = ref(''), retry = ref(0)
const revision = computed(() => props.dataset.revision || 1)
function searchPreview() { keyword.value = search.value; page.value = 1 }
function resetSearch() { search.value = ''; searchPreview() }
function loadValues(field, query, signal) {
  const params = new URLSearchParams({revision:revision.value, field, query})
  return api(`/datasets/${props.dataset.id}/preview-values?${params}`, {signal})
}
watch(conditions, () => { page.value = 1 }, {flush:'sync'})
watch([page, conditions, keyword, retry], async (_, __, onCleanup) => {
  const controller = new AbortController()
  onCleanup(() => controller.abort())
  loading.value = true
  error.value = ''
  try {
    const filters = Object.fromEntries(conditions.value.map(({key,values}) => [key,values]))
    const result = await api(`/datasets/${props.dataset.id}/preview?page=${page.value}&revision=${revision.value}`, {
      method:'POST', signal:controller.signal, body:JSON.stringify({filters, query:keyword.value}),
    })
    if (!controller.signal.aborted) preview.value = result
  } catch (e) {
    if (!controller.signal.aborted) { error.value = e.message; preview.value = {rows:[], total:0} }
  } finally {
    if (!controller.signal.aborted) loading.value = false
  }
}, {immediate:true})
</script>

<template>
  <section class="panel" aria-label="数据预览">
    <div class="panel-heading">
      <div><h2>{{dataset.name}} <span class="subtle">/ 数据预览 · 版本 {{revision}}</span></h2>
        <p class="muted">{{dataset.rows.toLocaleString()}} 行 · {{fields.length}} 个字段 · {{dataset.duplicates}} 条重复行</p>
      </div>
      <div class="preview-actions"><slot/></div>
    </div>
    <TableFilters v-model="drafts" :fields="fields" :load-values="loadValues" :has-search="!!(search || keyword)" @reset="resetSearch">
      <template #search><el-input v-model="search" placeholder="搜索数据内容，回车查询" :prefix-icon="Search" aria-label="搜索数据内容" clearable @keyup.enter="searchPreview" @clear="searchPreview"/></template>
    </TableFilters>
    <div v-if="error" role="alert">{{error}} <el-button link type="primary" @click="retry++">重试</el-button></div>
    <template v-else>
      <el-table v-loading="loading" :data="preview.rows" :max-height="theme.table==='fixed'?380:''" stripe empty-text="无匹配数据">
        <el-table-column v-for="c in dataset.columns" :key="c.name" :prop="c.name" :label="c.name" min-width="124" show-overflow-tooltip>
          <template #header><ColumnProfile :dataset-id="dataset.id" :revision="revision" :column="c.name" :theme-key="themeKey"/></template>
          <template #default="{row}">
            <span v-if="row[c.name] == null" class="muted" title="缺失值">—</span>
            <span v-else-if="row[c.name] === ''" class="muted">空字符串</span>
            <template v-else>{{row[c.name]}}</template>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination v-model:current-page="page" :total="preview.total" :page-size="20" :pager-count="5" layout="total,prev,pager,next"/>
    </template>
  </section>
</template>
