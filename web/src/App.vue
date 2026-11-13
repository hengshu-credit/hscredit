<script setup>
import {addFields} from './field-selection'
import FieldSelect from "./components/FieldSelect.vue"
import SamplePlan from './components/SamplePlan.vue'
import {useSamplePlan} from './sample-plan'
import TargetConfig from './components/TargetConfig.vue'
import {useTargetConfig} from './target-config'
import {computed,onMounted,onBeforeUnmount,provide,reactive,ref,watch} from 'vue'
import {ElMessage,ElMessageBox} from 'element-plus'
import {Coin,FolderOpened,DataAnalysis,Connection,Cpu,DocumentChecked,Setting,InfoFilled,Upload,ArrowRight,Refresh,List,Operation} from '@element-plus/icons-vue'
import {api} from './api'
import {useExploration,runningStatuses} from './exploration'
import {chosenFormat,columnKind,fieldChoices,pairChoices} from './explore-formats'
import {THEME_KEY,normalizeTheme,applyTheme} from './theme'
import ThemeSettings from './components/ThemeSettings.vue'
import DatasetCorrelation from './components/DatasetCorrelation.vue'
import TaskCenter from './components/TaskCenter.vue'
import TaskDetail from './components/TaskDetail.vue'
import {taskModules, taskUrl} from './task-center'
import PlatformSidebar from './components/PlatformSidebar.vue'
import {SIDEBAR_MIN_WIDTH,SIDEBAR_COMPACT_THRESHOLD,clampSidebarWidth} from './layout'
import PlatformFooter from './components/PlatformFooter.vue'
import DatasetProcessing from './components/DatasetProcessing.vue'
import DatasetSplit from './components/DatasetSplit.vue'
import DatasetImport from './components/DatasetImport.vue'
import DatasetSummary from './components/DatasetSummary.vue'
import {assetHierarchy,datasetHierarchy} from './split-editor'
import DatasetPreview from './components/DatasetPreview.vue'
import './processing.css'
import StrategyWorkbench from './components/StrategyWorkbench.vue'
import ModelWorkbench from './components/ModelWorkbench.vue'
import ServiceWorkbench from './components/ServiceWorkbench.vue'
import OptionalParameters from './components/OptionalParameters.vue'
import {scopeOptions} from './parameter-overrides'
import {option,named,binningOptions} from './workflow-parameters'
import InteractionExplorer from './components/InteractionExplorer.vue'
import ExploreFieldBoard from './components/ExploreFieldBoard.vue'
import BatchResultPicker from './components/BatchResultPicker.vue'
import WorkbookReport from './components/WorkbookReport.vue'
import {submitDatasetRuns} from './dataset-runs'

const menus=[{id:'datasets',name:'数据管理',icon:Coin,desc:'统一管理分析数据，建立可复用的风控数据资产。'}, {id:'explore',name:'数据探索',icon:DataAnalysis,desc:'从分布、缺失与相关性出发，理解每一个变量。'}, {id:'strategy',name:'策略挖掘',icon:Connection,desc:'探索变量分箱与高风险规则，形成可检验的策略假设。'}, {id:'model',name:'模型构建',icon:Cpu,desc:'训练模型，比较效果与解释预测。'}, {id:'services',name:'PaaS 服务',icon:Connection,desc:'发布模型与规则 API。'}, {id:'third_party',name:'三方测试',icon:DocumentChecked,desc:'评估三方变量价值，生成自动特征分析报告。'}, {id:'tasks',name:'任务管理',icon:List}]
const modelTab=ref('experiment'), page=ref('datasets')
const centerModule=ref('explore'), detailId=ref(''), thirdJobId=ref('')
const current=computed(()=>menus.find(m=>m.id===page.value)),datasets=ref([]),jobs=ref([]),catalog=ref({models:{}}),datasetId=ref(''),busy=ref(false),error=ref(''),drawer=ref(false)
provide('platformJobs',jobs)
const strategyContext=ref('strategy')
const sampleContext=computed(()=>page.value==='strategy'?strategyContext.value:page.value==='explore'?'summary':page.value)
const datasetTree=computed(()=>datasetHierarchy(datasets.value))
const assetTree=computed(()=>assetHierarchy(datasetTree.value))
const samplePlan=useSamplePlan(datasets,datasetId,sampleContext,assetTree)
const runDatasets=computed(()=>samplePlan.runDatasets)
let explicitlyEmpty=false
function selectPrimary(id){explicitlyEmpty=!id;datasetId.value=id}
let saved;try{saved=JSON.parse(localStorage.getItem(THEME_KEY)||'{}')}catch{saved={}}
const theme=reactive(normalizeTheme(saved)),themeKey=computed(()=>JSON.stringify(theme)),systemTheme=matchMedia('(prefers-color-scheme: dark)')
let savedTheme=normalizeTheme(saved)
provide('platformTheme',theme)
function persistTheme(){try{localStorage.setItem(THEME_KEY,JSON.stringify(theme));return true}catch{ElMessage.warning('浏览器无法保存主题，本次设置仍然生效');return false}}
watch(theme,()=>{applyTheme(theme);if(!drawer.value){savedTheme=normalizeTheme(theme);persistTheme()}},{deep:true,immediate:true})
function openTheme(){noteOpen.value=false;savedTheme=normalizeTheme(theme);drawer.value=true}
function cancelTheme(){Object.assign(theme,savedTheme);drawer.value=false}
function saveTheme(){savedTheme=normalizeTheme(theme);const saved=persistTheme();drawer.value=false;if(saved)ElMessage.success('主题已保存')}
const compactQuery=matchMedia('(max-width: 800px)'),compact=ref(compactQuery.matches)
const sidebarPreview=ref(null),mobileSidebarOpen=ref(false),noteOpen=ref(false)
const sidebarWidth=computed(()=>compact.value?(mobileSidebarOpen.value?theme.sidebarExpandedWidth:SIDEBAR_MIN_WIDTH):(sidebarPreview.value??theme.sidebarWidth))
const sidebarCollapsed=computed(()=>theme.layout==='left'&&sidebarWidth.value<SIDEBAR_COMPACT_THRESHOLD)
function resizeSidebar(width){
  const next=clampSidebarWidth(width)
  Object.assign(theme,{sidebarWidth:next,collapsed:next<SIDEBAR_COMPACT_THRESHOLD,...(next>=SIDEBAR_COMPACT_THRESHOLD?{sidebarExpandedWidth:next}:{})})
  sidebarPreview.value=null
}
function toggleSidebar(){noteOpen.value=false;if(compact.value)mobileSidebarOpen.value=!mobileSidebarOpen.value;else resizeSidebar(sidebarCollapsed.value?theme.sidebarExpandedWidth:SIDEBAR_MIN_WIDTH)}
function navigate(id){location.hash=id==='tasks'?taskUrl(centerModule.value):id;mobileSidebarOpen.value=false}
function openCenter(module,id){location.hash=taskUrl(module,id)}
function onCompact(){noteOpen.value=false;compact.value=compactQuery.matches;mobileSidebarOpen.value=false;sidebarPreview.value=null}
compactQuery.addEventListener('change',onCompact)
function onSystemTheme(){applyTheme(theme)}
systemTheme.addEventListener('change',onSystemTheme)
function onHash(){
  const [id,module,jobId]=location.hash.slice(1).split('/')
  noteOpen.value=false
  error.value=''
  if(id==='compare')modelTab.value='compare'
  page.value=id==='compare'?'model':menus.some(m=>m.id===id)?id:'datasets'
  if(page.value==='tasks'){
    centerModule.value=Object.hasOwn(taskModules,module)?module:'explore'
    detailId.value=jobId||''
  }
}
onHash()
window.addEventListener('hashchange',onHash)
const selected=computed(()=>datasets.value.find(d=>d.id===datasetId.value)),columns=computed(()=>selected.value?.columns||[])
provide('fieldPickerDataset',selected)
const numeric=computed(()=>columns.value.filter(c=>c.kind==='numeric')),dates=computed(()=>columns.value.filter(c=>c.kind==='date'))
const reportTarget=useTargetConfig(()=>selected.value,'report')
const summaryTarget=useTargetConfig(()=>selected.value,'explore')
const form=reactive({features:[],library_params:{},method:'quantile',max_n_bins:5,min_n_bins:1,min_bin_size:1,date:null,amount:null,freq:'M',pictures:['bin','ks'],corr:false,dropna:false,margins:true,long_format:false})
const availableFeatures=computed(()=>columns.value.filter(c=>!reportTarget.sources.includes(c.name)))
function selectAllFeatures(){
  const {names,overflow}=addFields(form.features,availableFeatures.value.map(c=>c.name),200)
  form.features=names
  if(overflow)ElMessage.info('分析特征上限为 200 个，另有 '+overflow+' 个字段未加入。')
}
watch(()=>reportTarget.sources.slice(),names=>{form.features=form.features.filter(c=>!names.includes(c))})
const summaryParams=ref({})
const axes=ref([]),chartGroup=ref(''),histBins=ref(20),timeFreq=ref('M')
const formats=reactive(new Map()),pairFormats=reactive(new Map())
const hasDate=computed(()=>columns.value.some(c=>axes.value.includes(c.name)&&c.kind==='date'))
const healthDataset=computed(()=>exploreJob.value
  ?datasets.value.find(d=>d.id===exploreJob.value.dataset_id&&d.revision===exploreJob.value.dataset_revision)
  :selected.value)
const missingCells=computed(()=>healthDataset.value?.columns.reduce((sum,c)=>sum+c.missing,0)||0)
const timeFrequencies={M:'按月',D:'按日',W:'按周（周日为边界）',Q:'按季度',Y:'按年',h:'按小时'}
function setExploreFields(next){
  axes.value=next.axes
  chartGroup.value=next.group
}
function kindOf(name){
  return eda.value?.interaction?.axes.find(a=>a.name===name)?.kind||columnKind(columns.value.find(c=>c.name===name))
}
const formatsFor=name=>fieldChoices(kindOf(name))
const combinations=computed(()=>axes.value.flatMap((x,i)=>axes.value.slice(i+1).map(y=>({key:JSON.stringify([x,y]),name:x+' × '+y,choices:pairChoices([kindOf(x),kindOf(y)])}))))
const stoppingJobs=ref([])
const exploreConfig=computed(()=>({dataset_id:datasetId.value,dataset_revision:selected.value?.revision||1,workflow:true,library_params:summaryParams.value,features:axes.value,feature:axes.value[0]||null,group:chartGroup.value||null,bins:histBins.value??20,freq:hasDate.value?timeFreq.value||'M':'M',...summaryTarget.payload,...(sampleContext.value==='summary'?samplePlan.payload:{})}))
const {eda,loading:edaLoading,following:followingExplore,runningCurrentConfig,jobId:exploreJobId,currentJob:exploreJob,view:viewExploreJob,chartsCurrent,load:submitExplore,settle,reset:resetExplore}=useExploration({config:exploreConfig,runDatasets,api,refresh,clearError:()=>{error.value=''},onError:message=>{error.value=message;ElMessage.error(message)}})
function loadExplore(){if(axes.value.length&&!summaryTarget.error)return submitExplore()}
const settleExplore=()=>settle(jobs.value)
watch([datasetId,()=>selected.value?.revision||1],()=>{
  summaryParams.value={};form.library_params={};noteOpen.value=false;thirdJobId.value='';resetExplore();timeFreq.value='M'
  form.date=selected.value?.default_date||dates.value[0]?.name||null;form.amount=selected.value?.default_amount||null
  form.features=selected.value?.default_features?.filter(c=>!reportTarget.sources.includes(c))||columns.value.filter(c=>!reportTarget.sources.includes(c.name)&&c.kind!=='date'&&!/(^id$|编号|编码|流水号|^MOB\d|DPD)/i.test(c.name)).map(c=>c.name).slice(0,30)
  axes.value=[numeric.value.find(c=>form.features.includes(c.name))?.name||columns.value[0]?.name].filter(Boolean)
  chartGroup.value='';formats.clear();pairFormats.clear()
  if(page.value==='datasets'&&datasetId.value)explicitlyEmpty=false
})
async function guard(fn){try{error.value='';return await fn()}catch(e){error.value=e.message;ElMessage.error(e.message)}}
async function refresh(){await guard(async()=>{
  const [d,j]=await Promise.all([api('/datasets'),api('/jobs')])
  datasets.value=d;jobs.value=j
  const available=new Set(d.map(item=>item.id))
  if(!available.has(datasetId.value)&&!explicitlyEmpty){
    datasetId.value=d.find(item=>!item.parent_id)?.id||d[0]?.id||''
  }
})}
async function stopJob(row){
  if(stoppingJobs.value.includes(row.id))return
  stoppingJobs.value.push(row.id)
  await guard(async()=>{const job=await api(`/jobs/${row.id}/stop`,{method:'POST'});ElMessage.info(job.status==='成功'?'任务已完成':job.message);await refresh();await settleExplore()})
  stoppingJobs.value=stoppingJobs.value.filter(id=>id!==row.id)
}

const splitOpen=ref(false),splitDataset=ref(null),splitPlanId=ref('')
const assetTable=ref(null)
function clickAsset(row){
  if(!row.isGroup)datasetId.value=row.id
}
function toggleAssetBranch(row,column,event){
  if(row.children.length&&!event.target.closest('button'))assetTable.value.toggleRowExpansion(row)
}
function openSplit(row,planId=''){
  splitDataset.value=row
  splitPlanId.value=row.group_path?row.splitting.plan_id:planId
  splitOpen.value=true
}
function editSplit(row){
  const source=datasets.value.find(d=>d.id===row.parent_id)
  if(source)openSplit(source,row.splitting.plan_id)
}
async function splitSaved(rows){
  const group=splitDataset.value
  await refresh()
  datasetId.value=group?.group_path&&datasets.value.some(d=>d.id===group.id)?group.id:rows[0].id
}
const copyingId=ref('')
async function copyAndSplit(row){
  if(copyingId.value)return
  copyingId.value=row.id
  try{
    await guard(async()=>{
      const copy=await api(`/datasets/${row.id}/copy`,{method:'POST',body:JSON.stringify({expected_revision:row.revision||1})})
      datasets.value=[copy,...datasets.value.filter(d=>d.id!==copy.id)]
      datasetId.value=copy.id
      openSplit(copy)
      ElMessage.success(`已创建「${copy.name}」`)
    })
  }finally{copyingId.value=''}
}
async function removeSplit(row){
  const {plan_id,plan_revision}=row.splitting
  const names=datasets.value.filter(d=>d.splitting?.plan_id===plan_id).map(d=>d.name)
  try{
    await ElMessageBox.confirm(`移除该方案及 ${names.length} 个子集（${names.join('、')}）？`,'删除划分方案',{confirmButtonText:'删除方案',cancelButtonText:'取消',type:'warning'})
  }catch{return}
  await guard(async()=>{
    await api(`/datasets/${row.parent_id}/split-plans/${plan_id}?plan_revision=${plan_revision}`,{method:'DELETE'})
    datasetId.value=row.parent_id
    await refresh()
  })
}
const processingOpen=ref(false),processingDataset=ref(null)
function openProcessing(row){processingDataset.value=row;processingOpen.value=true}
async function processingSaved(row){datasets.value=datasets.value.some(d=>d.id===row.id)?datasets.value.map(d=>d.id===row.id?row:d):[row,...datasets.value];datasetId.value=row.id;await refresh()}
const uploadOpen=ref(false)
const demoLoading=ref(false)
async function importDemo(){demoLoading.value=true;await guard(async()=>{const d=await api('/datasets/demo',{method:'POST'});await refresh();datasetId.value=d.id;ElMessage.success(`已加载完整样例数据：${d.rows.toLocaleString()} 行`)});demoLoading.value=false}
function datasetImported(dataset){
  datasets.value=[dataset,...datasets.value.filter(row=>row.id!==dataset.id)]
  selectPrimary(dataset.id)
  ElMessage.success(`已导入「${dataset.name}」，共 ${dataset.rows.toLocaleString()} 行`)
}
async function renameDataset(row){try{const {value}=await ElMessageBox.prompt('请输入数据集名称','重命名',{inputValue:row.name,inputValidator:v=>!!v.trim()||'名称不能为空',confirmButtonText:'保存',cancelButtonText:'取消'});await guard(async()=>{await api(`/datasets/${row.id}`,{method:'PATCH',body:JSON.stringify({name:value})});await refresh()})}catch{}}
async function deleteDataset(row){try{await ElMessageBox.confirm(`删除「${row.name}」及其数据文件？已生成的报告会保留。`,'删除数据集',{type:'warning',confirmButtonText:'删除',cancelButtonText:'取消'});await guard(async()=>{await api(`/datasets/${row.id}`,{method:'DELETE'});await refresh()})}catch{}}
const submitting=ref(false)
const activeCount=computed(()=>jobs.value.filter(j=>runningStatuses.includes(j.status)).length)
function registerJob(job){jobs.value=[job,...jobs.value.filter(j=>j.id!==job.id)]}
async function submit(){if(!datasetId.value||!form.features.length){ElMessage.warning('请选择数据集和分析特征');return}submitting.value=true;await guard(async()=>{
  if(reportTarget.error)throw new Error(reportTarget.error)
  const config={...form,...reportTarget.payload,kind:'third_party',dataset_id:datasetId.value,dataset_revision:selected.value.revision||1}
  const created=await submitDatasetRuns(api,config,runDatasets.value)
  if(datasetId.value===config.dataset_id && (selected.value?.revision||1)===config.dataset_revision)thirdJobId.value=created[0].id
  ElMessage.success(`已提交 ${created.length} 个三方测试任务`);await refresh()
});submitting.value=false}
const exploreOptions=computed(()=>[
  option('bins','分布格数','用于直方图和交互密度统计',20,{group:'分布统计',min:2,max:100}),
  ...(hasDate.value?[option('freq','时间汇总频率','按有效日期聚合，可输入 2W 等频率','M',{group:'分布统计',choices:named(timeFrequencies),custom:true})]:[]),
])
const reportOptions=computed(()=>[
  option('date','日期字段','按期查看分布',selected.value?.default_date||dates.value[0]?.name||null,{fields:dates.value}),
  option('amount','金额字段','增加金额口径报告',selected.value?.default_amount||null,{fields:numeric.value}),
  option('freq','时间粒度','日期分组的汇总频率','M',{choices:named({D:'日',W:'周',M:'月',Q:'季度'}),when:d=>!!d.date}),
  ...binningOptions('分箱与规则',catalog.value.parameters?.methods),
  option('pictures','报告图表','选择 Excel 中包含的图表',['bin','ks'],{hint:'分箱图、KS 曲线',multi:true,choices:named({bin:'分箱图',ks:'KS 曲线',hist:'分布直方图'})}),
  option('corr','特征相关性','展示相关系数',false),option('dropna','剔除特征缺失','计算时移除特征缺失行',false),
  option('margins','分箱合计行','在分箱表末尾显示合计',true),option('long_format','长表输出','按标签分行展示',false),
])
let timer,polling=false
watch(page, value => {
  if (value==='tasks') guard(async()=>{jobs.value=await api('/jobs')})
})
onMounted(async()=>{busy.value=true;await guard(async()=>{catalog.value=await api('/catalog')});await refresh();busy.value=false;timer=setInterval(async()=>{if(polling)return;polling=true;try{jobs.value=await api('/jobs');if(error.value.startsWith('状态刷新失败：'))error.value='';await settleExplore()}catch(e){error.value=`状态刷新失败：${e.message}`}finally{polling=false}},2500)})
onBeforeUnmount(()=>{clearInterval(timer);compactQuery.removeEventListener('change',onCompact);systemTheme.removeEventListener('change',onSystemTheme);window.removeEventListener('hashchange',onHash)})
</script>

<template>
  <div class="shell" @keydown.esc="mobileSidebarOpen=false" :class="{'top-layout':theme.layout==='top',collapsed:sidebarCollapsed,'mobile-sidebar-open':compact&&mobileSidebarOpen}" :style="{'--sidebar-width':sidebarWidth+'px','--sidebar-space':(compact?64:sidebarWidth)+'px'}">
    <PlatformSidebar :menus="menus" :active="page" :width="sidebarWidth" :collapsed="sidebarCollapsed" :resizable="!compact&&theme.layout==='left'" :top="theme.layout==='top'" @navigate="navigate" @toggle="toggleSidebar" @resize="sidebarPreview=$event" @resize-end="resizeSidebar"/>
    <button v-if="compact&&mobileSidebarOpen&&theme.layout==='left'" class="sidebar-backdrop" aria-label="关闭导航" @click="mobileSidebarOpen=false"/>
    <div class="workspace">
      <header class="topbar">
        <nav class="header-breadcrumb" aria-label="页面路径"><ol><li><el-icon><component :is="current.icon"/></el-icon><h1>{{current.name}}</h1></li></ol></nav>
        <div class="header-right">
          <el-button v-if="taskModules[page]" :icon="List" text @click="openCenter(page)">查看历史任务</el-button>
          <el-button v-if="page!=='datasets'" :icon="Refresh" text @click="refresh">刷新任务</el-button>
          <el-tag type="info" effect="plain">{{activeCount?activeCount+' 个任务进行中':'工作空间就绪'}}</el-tag><el-button :icon="Setting" text @click="openTheme">主题配置</el-button>
        </div>
      </header>
      <main v-loading="busy && page!=='explore'" :aria-label="current.name">
        <el-alert v-if="error" :title="error" type="error" show-icon closable @close="error=''" class="error-alert"/>
        <template v-if="page==='datasets'">
          <div class="stat-grid"><div class="stat-card"><span>已接入数据集</span><strong>{{datasetTree.length}}<small>个</small></strong><el-icon><Coin/></el-icon></div><div class="stat-card"><span>数据总量</span><strong>{{datasetTree.reduce((n,d)=>n+d.rows,0).toLocaleString()}}<small>行</small></strong><el-icon><DataAnalysis/></el-icon></div><div class="stat-card"><span>已完成分析</span><strong>{{jobs.filter(j=>j.status==='成功').length}}<small>次</small></strong><el-icon><DocumentChecked/></el-icon></div></div>
          <section class="panel">
            <div class="panel-heading asset-heading">
              <div class="panel-heading-tools">
                <h2>数据资产</h2>
                <span class="muted">CSV / Excel / Parquet / 数据库</span>
                <el-popover v-if="selected?.note" v-model:visible="noteOpen" placement="bottom-end" :width="compact?280:420" trigger="click" :content="selected.note">
                  <template #reference><el-button class="dataset-note-trigger" text :icon="InfoFilled">数据说明</el-button></template>
                </el-popover>
              </div>
              <div class="asset-actions">
                <el-button :loading="demoLoading" @click="importDemo">加载 hscredit 示例</el-button>
                <el-button type="primary" :icon="Upload" @click="uploadOpen=true">导入数据集</el-button>
              </div>
            </div>
            <el-empty v-if="!datasets.length" description="导入第一份数据，开始风控分析"><el-button type="primary" @click="uploadOpen=true">导入数据集</el-button></el-empty>
            <el-table v-else ref="assetTable" :data="assetTree" default-expand-all :indent="24" stripe :max-height="theme.table==='fixed'?420:''" @row-click="clickAsset" @row-dblclick="toggleAssetBranch" row-key="id" :row-class-name="({row})=>row.id===datasetId?'current-row':''">
              <el-table-column label="数据集名称" min-width="240"><template #default="{row}">
                <span class="dataset-name" :class="{'dataset-group':row.isGroup||row.group_path}" :title="row.children.length?(row.isGroup?'双击展开或折叠':'单击预览，双击展开或折叠'):undefined"><el-icon><FolderOpened v-if="row.isGroup||row.group_path"/><Coin v-else/></el-icon>{{row.name}}</span>
              </template></el-table-column>
              <el-table-column prop="rows" label="样本量" width="96" sortable/>
              <el-table-column label="字段数" width="80"><template #default="{row}"><template v-if="!row.isGroup">{{row.columns.length}}</template></template></el-table-column>
              <el-table-column label="缺失比例" width="96"><template #default="{row}"><template v-if="!row.isGroup">{{(row.missing_rate*100).toFixed(2)}}%</template></template></el-table-column>
              <el-table-column label="导入时间" min-width="160"><template #default="{row}"><template v-if="!row.isGroup">{{new Date(row.created_at).toLocaleString()}}</template></template></el-table-column>
              <el-table-column label="操作" width="224" :fixed="compact?undefined:'right'"><template #default="{row}">
                <template v-if="!row.isGroup">
                  <el-button v-if="row.parent_id&&!row.group_path" link type="primary" title="调整生成此子集的方案" @click.stop="editSplit(row)">重划</el-button>
                  <el-button link type="primary" :title="row.group_path?'调整分组并更新整体方案':row.parent_id?'划分此子集':'配置数据集划分'" @click.stop="openSplit(row)">{{row.parent_id&&!row.group_path?'细分':'划分'}}</el-button>
                  <el-button v-if="!row.parent_id" link type="primary" title="复制数据集并配置新划分" :loading="copyingId===row.id" :disabled="!!copyingId" @click.stop="copyAndSplit(row)">复制</el-button>
                  <el-button link type="primary" title="加工字段" @click.stop="openProcessing(row)">加工</el-button>
                  <el-button v-if="!row.group_path" link type="primary" title="修改数据集名称" @click.stop="renameDataset(row)">改名</el-button>
                  <el-button v-if="row.parent_id&&!row.group_path" link type="danger" title="删除所属划分方案" @click.stop="removeSplit(row)">删除</el-button>
                  <el-button v-else-if="!row.parent_id" link type="danger" title="删除数据集" @click.stop="deleteDataset(row)">删除</el-button>
                </template>
              </template></el-table-column>
            </el-table></section>
          <DatasetPreview v-if="selected" :key="`${selected.id}:${selected.revision||1}`" :dataset="selected" :theme-key="themeKey">
            <el-button v-if="selected.parent_id&&!selected.group_path" title="调整所属划分方案" @click="editSplit(selected)">重划</el-button>
            <el-button v-else @click="openSplit(selected)">划分</el-button>
            <el-button :icon="Operation" title="加工字段" @click="openProcessing(selected)">加工</el-button>
          </DatasetPreview>
          <DatasetSummary v-if="selected" :dataset="selected"/>
          <DatasetCorrelation v-if="selected" :key="`correlation:${selected.id}:${selected.revision||1}`" :dataset="selected" :theme-key="themeKey"/>
        </template>
        <TaskCenter v-else-if="page==='tasks'" :module="centerModule" :id="detailId" :jobs="jobs" :datasets="datasets" :catalog="catalog.models" :parameter-catalog="catalog.parameters" :stopping="stoppingJobs" @stop="stopJob" @created="registerJob"/>
        <template v-else>
          <SamplePlan :compact="page==='explore'" v-if="page!=='services'" :context="sampleContext" :primary="datasetId" :plan="samplePlan" :tree="assetTree" @primary="selectPrimary"><el-popover v-if="selected?.note" v-model:visible="noteOpen" placement="bottom-end" :width="compact?280:420" trigger="click" :content="selected.note"><template #reference><el-button class="dataset-note-trigger" text :icon="InfoFilled">数据说明</el-button></template></el-popover><el-button link type="primary" @click="navigate('datasets')">管理数据集 <el-icon><ArrowRight/></el-icon></el-button></SamplePlan>
          <ServiceWorkbench v-if="page==='services'" :dataset="selected" :datasets="datasets"/>
          <p v-else-if="busy && page==='explore' && !selected" class="muted" role="status">正在读取数据集…</p>
          <el-empty v-else-if="!selected"  :description="datasets.length?'请选择用于本次分析的主样本':'请先导入数据集'"><el-button v-if="!datasets.length" type="primary" @click="navigate('datasets');uploadOpen=true">导入数据集</el-button></el-empty>
          <StrategyWorkbench :catalog="catalog.parameters" :sample-plan="samplePlan" @sample-context="strategyContext=$event" :key="selected.id+':'+selected.revision" v-else-if="page==='strategy'" :dataset="selected" :datasets="datasets" :run-datasets="runDatasets"/>
          <template v-else-if="page==='model'"><el-tabs v-model="modelTab" class="workbench-tabs"><el-tab-pane label="模型实验配置" name="experiment"/><el-tab-pane label="模型效果对比" name="compare"/></el-tabs>
            <ModelWorkbench :sample-plan="samplePlan" :key="selected.id+':'+selected.revision+modelTab" :dataset="selected" :datasets="datasets" :run-datasets="runDatasets" :catalog="catalog" :compare="modelTab==='compare'"/>
          </template>
          <template v-else-if="page==='explore'">
            <ExploreFieldBoard :key="selected.id+':'+(selected.revision||1)" :columns="columns" :axes="axes" :group="chartGroup" @change="setExploreFields">
              <template #format="{field}"><select class="shelf-format" :aria-label="field+'的分析图形'" :value="chosenFormat(formatsFor(field),formats.get(field))" @change="formats.set(field,$event.target.value)"><option v-for="[id,label] in formatsFor(field)" :key="id" :value="id">{{label}}</option></select></template>
              <template #combinations><div v-if="combinations.length" class="combination-formats"><strong>交叉图形</strong><label v-for="p in combinations" :key="p.key">{{p.name}} <select :aria-label="p.name+'的布局图形'" :value="chosenFormat(p.choices,pairFormats.get(p.key))" @change="pairFormats.set(p.key,$event.target.value)"><option v-for="[id,label] in p.choices" :key="id" :value="id">{{label}}</option></select></label></div></template>
              <template #actions>
                <el-button v-if="followingExplore&&exploreJobId" @click="stopJob({id:exploreJobId})">停止</el-button>
                <el-button type="primary" :loading="runningCurrentConfig" :disabled="!axes.length||!!summaryTarget.error" @click="loadExplore">运行分析</el-button>
              </template>
              <template #settings>
                <TargetConfig :config="summaryTarget" :dataset="selected"/>
                <OptionalParameters context="explore" title="分布统计" :specs="exploreOptions" :extensions="scopeOptions(catalog.parameters,'explore')" v-model:overrides="summaryParams" :values="{bins:histBins,freq:timeFreq}" @change="(k,v)=>k==='bins'?histBins=v:timeFreq=v"/>
              </template>
              <BatchResultPicker :job="exploreJob" @select="viewExploreJob"/>

              <el-alert v-if="eda&&!chartsCurrent&&!edaLoading" title="字段或设置已更改，点击「运行分析」更新下方结果。" type="info" :closable="false" class="explore-stale"/>
              <div v-if="!eda" class="panel board-placeholder" role="status"><el-icon><DataAnalysis/></el-icon>{{edaLoading?'正在生成分析结果…':axes.length?'点击「运行分析」查看所选字段的分布。':'将字段放入分析位置后，在这里查看分布图。'}}</div>
              <InteractionExplorer v-if="eda?.interaction" :key="eda.job.id" :result="eda.interaction" :summary="eda" :job="eda.job" :theme-key="themeKey" :formats="formats" :pair-formats="pairFormats" @format="(name,value)=>formats.set(name,value)" @pair-format="(key,value)=>pairFormats.set(key,value)"/>
            <details v-if="healthDataset" class="panel dataset-health"><summary>全表数据检查 · {{healthDataset.name}}</summary><div class="dataset-health-metrics"><div><strong>{{((healthDataset.missing_rate||0)*100).toFixed(2)}}<small>%</small></strong><span>缺失单元格比例</span><p class="muted">{{missingCells.toLocaleString()}} 个缺失单元格 ÷ {{(healthDataset.rows*healthDataset.columns.length).toLocaleString()}} 个总单元格。可在数据管理的字段概况中查看各字段的缺失数。</p></div><div><strong>{{healthDataset.duplicates.toLocaleString()}}</strong><span>完全重复行数</span><p class="muted">所有字段均相同的额外行数，首次出现不计入；客户编号重复不等于整行重复。</p></div></div></details>
              <WorkbookReport v-if="exploreJob?.status==='成功'" :key="exploreJob.id" :job="exploreJob" foldable/>
            </ExploreFieldBoard>
          </template>
          <template v-else>
            <section class="panel config-panel"><div class="panel-heading"><h2><el-icon><Operation/></el-icon> 三方评估配置</h2><el-tag effect="plain">字段随数据集联动</el-tag></div>
              <el-form label-position="top">
                <TargetConfig :config="reportTarget" :dataset="selected"/>
                <div class="form-grid">
                  <el-form-item label="分析特征" class="full-width" required><div class="feature-field"><FieldSelect v-model="form.features" :fields="availableFeatures" multiple :limit="200" label="分析特征" placeholder="选择参与分析的字段"/><el-button @click="selectAllFeatures">全选</el-button><el-button @click="form.features=[]">清空</el-button></div><span class="form-help">已选择 {{form.features.length}} 个特征。标签已排除；请主动排除客户标识和放款后产生的字段。</span></el-form-item>
                </div>
                <OptionalParameters context="third_party" :key="selected.id+':'+selected.revision" :specs="reportOptions" :extensions="scopeOptions(catalog.parameters,'third_party')" v-model:overrides="form.library_params" :values="form" @change="(k,v)=>form[k]=v"/>
                <div class="submit-bar"><span class="muted">{{runDatasets.length>1?runDatasets.length+' 份样本':selected.name}} · {{form.features.length}} 个特征</span><el-button type="primary" size="large" :loading="submitting" :disabled="!form.features.length||!!reportTarget.error" @click="submit">生成三方测试报告 <el-icon><ArrowRight/></el-icon></el-button></div>
              </el-form>
            </section>
          </template>
          <TaskDetail v-if="page==='third_party'&&thirdJobId" embedded :key="thirdJobId" :id="thirdJobId" :jobs="jobs" :stopping="stoppingJobs" @stop="stopJob" @open="job=>openCenter('third_party',job.id)" @select="thirdJobId=$event.id"/>
        </template>
        <PlatformFooter/>
      </main>
    </div>
  </div>
  <DatasetSplit v-model="splitOpen" :dataset="splitDataset" :datasets="datasets" :plan-id="splitPlanId" @saved="splitSaved"/>
  <DatasetProcessing v-model="processingOpen" :dataset="processingDataset" :catalog="catalog.processing||{functions:[],operators:[]}" @saved="processingSaved"/>
  <DatasetImport v-model="uploadOpen" @imported="datasetImported"/>
  <ThemeSettings v-model="drawer" :theme="theme" @preview="Object.assign(theme,$event)" @save="saveTheme" @cancel="cancelTheme"/>
</template>
