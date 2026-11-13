<script setup>
import {computed, inject, ref, shallowRef, watch} from 'vue'
import {InfoFilled} from '@element-plus/icons-vue'
import {api} from '../api'
import Chart from './Chart.vue'
import FieldSelect from './FieldSelect.vue'

const props = defineProps({dataset:{type:Object,required:true}, themeKey:String})
const theme = inject('platformTheme')
// 刷新资产列表会替换元数据对象；相同字段顺序无需重置当前页。
const fieldNames = computed(previous => {
  const names = props.dataset.columns.map(field => field.name)
  return previous?.length === names.length && names.every((name,i) => previous[i] === name) ? previous : names
})
const kinds = computed(() => new Map(props.dataset.columns.map(field => [field.name,field.kind])))
const rows = ref([]), separateColumns = ref(null)
const separate = ref(false), view = ref('chart'), pageSize = ref(16)
const rowPage = ref(1), columnPage = ref(1), retry = ref(0)
const rowFields = computed(() => rows.value.length ? rows.value : fieldNames.value)
const columnFields = computed(() => separate.value && separateColumns.value !== null
  ? (separateColumns.value.length ? separateColumns.value : fieldNames.value) : rowFields.value)
function changeMode(value) {
  if (value && separateColumns.value === null) separateColumns.value = [...rows.value]
  separate.value = value
}
watch([rowFields,pageSize], () => { rowPage.value = 1 })
watch([columnFields,pageSize], () => { columnPage.value = 1 })
function page(fields, number) {
  return fields.slice((number - 1) * pageSize.value, number * pageSize.value)
}
const selection = computed(() => ({
  rows:page(rowFields.value,rowPage.value), columns:page(columnFields.value,columnPage.value),
}))
const pageCapacity = computed(() => Math.min(pageSize.value,rowFields.value.length))
const result = shallowRef(null), loading = ref(false), error = ref('')
const statusText = computed(() => {
  if (error.value) return result.value ? '更新失败，显示上次结果' : '计算失败'
  if (loading.value) return result.value ? '正在更新，显示上次结果' : '正在计算字段相关性…'
  if (!result.value) return '暂无可分析字段'
  return `当前窗口 ${result.value.rows.length} 行 × ${result.value.columns.length} 列`
})
watch([selection,retry], async ([fields],_,onCleanup) => {
  const controller = new AbortController()
  onCleanup(() => controller.abort())
  error.value = ''
  if (!fields.rows.length || !fields.columns.length) {
    result.value = null
    loading.value = false
    return
  }
  loading.value = true
  const capacity = pageCapacity.value
  try {
    const matrix = await api(`/datasets/${props.dataset.id}/correlation?revision=${props.dataset.revision || 1}`, {
      method:'POST', body:JSON.stringify(fields), signal:controller.signal,
    })
    if (!controller.signal.aborted) result.value = {...matrix,capacity}
  } catch (e) {
    if (!controller.signal.aborted) error.value = e.message
  } finally {
    if (!controller.signal.aborted) loading.value = false
  }
}, {immediate:true})

// 按一页可容纳的行数占位，末页行数较少时分页按钮仍在原位。
const windowRows = computed(() => result.value?.capacity ?? pageCapacity.value)
const chartHeight = computed(() => Math.max(220,140+windowRows.value*24))
const windowStyle = computed(() => {
  const tableHeight = `calc(var(--table-header-height) + ${windowRows.value} * var(--table-row-height) + 4px)`
  return {
    '--matrix-table-height':theme.table === 'fixed' ? `min(420px, ${tableHeight})` : tableHeight,
    '--matrix-chart-height':`${Math.min(620,chartHeight.value)}px`,
  }
})

function method(a,b) {
  const count = Number(kinds.value.get(a) === 'category') + Number(kinds.value.get(b) === 'category')
  return ['Pearson r','相关比 η','Cramér’s V'][count]
}
const display = value => value === null ? '—' : value.toFixed(4)
const tableRows = computed(() => result.value?.rows.map((name,i) => ({name,values:result.value.values[i]})) || [])
const chart = computed(() => {
  const {rows,columns,values} = result.value
  const data = []
  for (const [i,row] of values.entries()) {
    row.forEach((value,j) => { if (value !== null) data.push([j,i,value]) })
  }
  return {
    animation:false,
    tooltip:{trigger:'item',formatter:({value:[j,i,v]}) => `${rows[i]} × ${columns[j]}\n${method(rows[i],columns[j])}  ${display(v)}`},
    grid:{left:8,right:16,top:10,bottom:46,containLabel:true},
    xAxis:{type:'category',data:columns,axisLabel:{rotate:columns.length>4?35:0,width:110,overflow:'truncate',hideOverlap:true}},
    yAxis:{type:'category',inverse:true,data:rows,axisLabel:{width:110,overflow:'truncate',hideOverlap:true}},
    visualMap:{min:-1,max:1,orient:'horizontal',left:'center',bottom:0,text:['+1','−1']},
    series:[{type:'heatmap',data}],
  }
})
</script>

<template>
  <section class="panel correlation-panel" aria-label="字段相关性">
    <div class="panel-heading">
      <h2>字段相关性 <span class="subtle">/ {{dataset.name}}</span></h2>
      <el-popover placement="bottom-end" trigger="click" :width="280">
        <template #reference><el-button link :icon="InfoFilled">计算说明</el-button></template>
        每对字段使用共同有效的样本。数值之间用 Pearson r，类别之间用 Cramér’s V，类别与数值之间用相关比 η。日期按时间先后、布尔值按 0/1 计算。常量或有效样本不足时，图中留空，表中显示「—」。
      </el-popover>
    </div>
    <div class="matrix-controls">
      <div class="matrix-field">
        <span>{{separate?'行字段':'分析字段'}}</span>
        <FieldSelect v-model="rows" :fields="dataset.columns" multiple :label="separate?'行字段':'分析字段'"
          :placeholder="`全部 ${fieldNames.length} 个字段 · 点击筛选`"/>
      </div>
      <div v-if="separate" class="matrix-field">
        <span>列字段</span>
        <FieldSelect v-model="separateColumns" :fields="dataset.columns" multiple label="列字段"
          :placeholder="`全部 ${fieldNames.length} 个字段 · 点击筛选`"/>
      </div>
      <el-checkbox class="option-toggle" :model-value="separate" @update:model-value="changeMode">单独选择列字段</el-checkbox>
    </div>
    <div class="matrix-heading">
      <el-radio-group v-model="view" size="small" aria-label="相关性视图">
        <el-radio-button value="chart">热力图</el-radio-button>
        <el-radio-button value="table">相关性表</el-radio-button>
      </el-radio-group>
      <span class="muted" role="status">{{statusText}}</span>
    </div>
    <div v-if="windowRows" class="matrix-window" :data-view="view" :style="windowStyle" :aria-busy="loading">
      <div v-if="result && view==='chart'" class="matrix-chart" tabindex="0">
        <Chart :option="chart" :theme-key="themeKey" aria-label="字段相关性热力图"
          :style="{height:chartHeight+'px',minWidth:(128+result.columns.length*42)+'px'}"/>
      </div>
      <el-table v-else-if="result" :data="tableRows" height="var(--matrix-table-height)" stripe aria-label="相关性表">
        <el-table-column prop="name" label="字段" fixed width="136" show-overflow-tooltip/>
        <el-table-column v-for="(name,j) in result.columns" :key="j" :label="name" align="right" min-width="106" show-overflow-tooltip>
          <template #default="{row}"><span :title="method(row.name,name)">{{display(row.values[j])}}</span></template>
        </el-table-column>
      </el-table>
      <div v-if="error" class="matrix-error" role="alert">
        {{error}} <el-button link type="primary" @click="retry++">重试</el-button>
      </div>
    </div>
    <div v-if="Math.max(rowFields.length,columnFields.length)>16" class="matrix-pages">
      <el-select v-model="pageSize" size="small" aria-label="每页字段数">
        <el-option v-for="size in [16,32,64]" :key="size" :value="size" :label="size+' 个字段/页'"/>
      </el-select>
      <div v-if="rowFields.length>pageSize" class="matrix-axis-page">行页
        <el-pagination v-model:current-page="rowPage" small :total="rowFields.length" :page-size="pageSize" :pager-count="5" layout="prev,pager,next"/>
      </div>
      <div v-if="columnFields.length>pageSize" class="matrix-axis-page">列页
        <el-pagination v-model:current-page="columnPage" small :total="columnFields.length" :page-size="pageSize" :pager-count="5" layout="prev,pager,next"/>
      </div>
    </div>
  </section>
</template>

<style scoped>
.matrix-controls,.matrix-heading,.matrix-pages,.matrix-axis-page{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.matrix-controls{background:var(--soft);border:1px solid var(--line-soft);padding:10px 12px;gap:12px 20px}
.matrix-field{display:flex;align-items:center;gap:10px;flex:1 1 230px;min-width:0}
.matrix-field>span{flex:none;font-size:12px;color:var(--muted)}
.matrix-controls .el-checkbox{margin:0}
.matrix-heading{display:grid;grid-template-columns:auto 1fr;margin:14px 0 10px}
.matrix-heading [role=status]{text-align:right;font-size:12px}
.matrix-window{position:relative;min-width:0;min-height:var(--matrix-chart-height)}
.matrix-window[data-view=table]{min-height:var(--matrix-table-height)}
.matrix-chart{height:var(--matrix-chart-height);overflow:auto;overscroll-behavior:contain}
.matrix-error{position:absolute;inset:0 0 auto;z-index:4;display:flex;justify-content:center;align-items:center;gap:8px;padding:6px 10px;background:var(--danger-bg);color:var(--danger);overflow-wrap:anywhere}
.matrix-pages{justify-content:flex-end;gap:8px 20px;margin-top:12px;color:var(--muted);font-size:12px}
.matrix-pages .el-select{width:132px}.matrix-pages .el-pagination{margin:0}
.correlation-panel .el-table{font-variant-numeric:tabular-nums}
@media(max-width:640px){.matrix-field{flex-basis:100%}.matrix-pages{justify-content:flex-start}}
</style>
