<script setup>
import TableFrame from './TableFrame.vue'
import {computed,ref,watch} from 'vue'
import FeatureSection from './FeatureSection.vue'
import DataTable from './DataTable.vue'
import ReportImage from './ReportImage.vue'
import WorkbookReport from './WorkbookReport.vue'
import {download} from '../api'
import {ElMessage} from 'element-plus'
const props=defineProps({job:Object,selectable:Boolean})
const emit=defineEmits(['select'])
const chosen=ref([])
const result=computed(()=>props.job?.result||{})
const tables=computed(()=>Object.entries(result.value.tables||{}).filter(([name])=>!name.startsWith('报告预览')))
const comparisons=computed(()=>tables.value.filter(([name])=>name.startsWith('跨样本指标 / ')))
const comparisonName=ref('')
watch(comparisons,items=>{comparisonName.value=items[0]?.[0]||''},{immediate:true})
const candidates=computed(()=>[...(result.value.candidates||[]),...(result.value.branches||[])])
const dataset=ref(''),tableName=ref('')
watch(()=>props.job?.id,()=>{chosen.value=[]})
const datasets=computed(()=>result.value.datasets||Object.keys(candidates.value[0]?.metrics||{}))
watch(datasets,names=>{dataset.value=names[0]||''},{immediate:true})
function matchesDataset(name){
  const owner=datasets.value.find(d=>name.startsWith(d+' / ')||name.endsWith(' / '+d))
  return !owner || owner===dataset.value
}
const relevantTables=computed(()=>tables.value.filter(([name])=>!name.startsWith('候选规则 / ')&&!name.startsWith('跨样本指标 / ')&&matchesDataset(name)))
watch(relevantTables,items=>{tableName.value=items[0]?.[0]||''},{immediate:true})
const workbook=computed(()=>result.value.workbooks?.[dataset.value])
const downloads=computed(()=> (props.job?.artifacts||[]).filter(file=>!file.endsWith('.png')&&!file.endsWith('.xlsx')))
const activeRows=computed(()=>result.value.tables?.[tableName.value]||[])
const primaryTable=computed(()=>relevantTables.value.find(([name])=>name===dataset.value+' / 2D分箱表'||name.startsWith('树规则表')&&(!name.includes(' / ')||name.endsWith(' / '+dataset.value)))?.[0])
const images=computed(()=>[...(result.value.images||[]),...Object.values(result.value.visuals||{}).flatMap(v=>(v.images||[]).map(img=>({...img,title:v.name+' / '+img.title})))])
async function save(file){try{await download(props.job.id,file)}catch(e){ElMessage.error(e.message)}}
</script>
<template>
  <section v-if="job" class="panel workflow-results">
    <div class="panel-heading"><div><h2>分析结果</h2><span class="muted">{{job.dataset_name}} · 版本 {{job.dataset_revision||1}}</span></div><el-tag>{{job.status}}</el-tag></div>
    <el-alert :title="job.message" :type="job.status==='失败'?'error':'info'" :closable="false"/>
    <el-progress v-if="['运行中','排队中','停止中'].includes(job.status)" :percentage="job.progress"/>
    <p v-if="result.note" class="muted">{{result.note}}</p>
    <template v-if="job.status==='成功'">
      <div v-if="datasets.length>1" class="result-sample"><label>查看评估样本</label><el-select v-model="dataset" aria-label="查看评估样本"><el-option v-for="label in datasets" :key="label" :value="label"/></el-select></div>
      <WorkbookReport :key="job.id" :job="job" :preferred-file="workbook" foldable/>
      <section v-if="comparisons.length">
        <div class="panel-heading"><h3>跨样本指标</h3><el-select v-if="comparisons.length>1" v-model="comparisonName" aria-label="选择模型和评估标签" style="width:min(320px,100%)">
          <el-option v-for="([name]) in comparisons" :key="name" :value="name" :label="name.slice('跨样本指标 / '.length)"/>
        </el-select><span v-else class="muted">{{comparisonName.slice('跨样本指标 / '.length)}}</span></div>
        <DataTable :rows="result.tables[comparisonName]||[]"/>
      </section>
      <section v-if="selectable&&result.candidates?.length">
        <h3>{{result.strategy_mode==='pair'?'二维分箱规则表':['binary','multiway'].includes(result.strategy_mode)?'树节点规则表':'单变量分箱规则表'}}</h3>
        <p class="muted">名称可修改；多选后按“或”合并为一条规则，也可逐条加入规则集。切换数据集可核对同一规则的命中表现。</p>
        <TableFrame max-height="420" v-slot="{tableSize}">
          <el-table v-bind="tableSize" :data="candidates" row-key="id" @selection-change="chosen=$event">
            <el-table-column type="selection" width="40"/>
            <el-table-column label="规则名称" min-width="176"><template #default="{row}"><el-input v-model="row.name" aria-label="规则名称"/></template></el-table-column>
            <el-table-column v-for="metric in ['样本总数','样本占比','坏样本率','LIFT值']" :key="metric" :label="metric==='样本总数'?'命中样本数':metric" min-width="92"><template #default="{row}">{{row.metrics?.[dataset]?.[metric]==null?'—':Number(row.metrics[dataset][metric].toPrecision(5))}}</template></el-table-column>
            <el-table-column prop="expression" label="精确执行表达式" min-width="272"/>
          </el-table>
        </TableFrame>
        <div class="download-bar"><el-button :disabled="!chosen.length" @click="emit('select',chosen.map(r=>({...r})))">逐条加入规则集</el-button><el-button :disabled="!chosen.length" @click="emit('select',[{name:chosen.map(r=>r.name).join(' 或 ').slice(0,100),expression:chosen.map(r=>'('+r.expression+')').join(' | ')}])">合并选中分箱并命名</el-button></div>
      </section>
      <section v-if="primaryTable"><h3>{{primaryTable}}</h3><DataTable :rows="result.tables[primaryTable]"/><div class="report-visual-grid"><ReportImage v-for="img in images.filter(i=>i.title==='树图'||i.title===dataset+' / 2D分箱图')" :key="img.file" :job="job.id" :file="img.file" :title="img.title"/></div></section>
      <section v-if="result.features?.length"><h3>各变量分箱与稳定性</h3><div :class="{'feature-pair':result.strategy_mode==='pair'}"><FeatureSection v-for="feature in result.features" :key="feature" :feature="feature" :job="job.id" :dataset="dataset" :tables="tables" :images="images"/></div></section>
      <el-tabs><el-tab-pane v-if="relevantTables.length" label="评估表格" lazy><el-select v-model="tableName" filterable aria-label="选择评估表格"><el-option v-for="([name]) in relevantTables" :key="name" :value="name"/></el-select><DataTable :rows="activeRows"/></el-tab-pane>
        <el-tab-pane label="分布、趋势与解释图" lazy><div class="report-visual-grid"><ReportImage v-for="img in images.filter(i=>!i.title.startsWith('报告预览')&&matchesDataset(i.title))" :key="img.file" :job="job.id" :file="img.file" :title="img.title"/></div></el-tab-pane></el-tabs>
      <div v-if="downloads.length" class="download-bar"><el-button v-for="file in downloads" :key="file" @click="save(file)">下载 {{file}}</el-button></div>
    </template>
  </section>
</template>
<style scoped>
.result-sample{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:14px 0}.result-sample .el-select{width:min(360px,100%)}
.feature-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.feature-pair>section{min-width:0}@media(max-width:900px){.feature-pair{grid-template-columns:1fr}}
</style>
