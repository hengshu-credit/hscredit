<script setup>
import TargetConfig from './TargetConfig.vue'
import {copyTargetDraft,useTargetConfig} from '../target-config'
import {datasetDraftKey,rememberForm} from '../form-drafts'
import {modelResultSelection,followModelResult} from '../model-result'
import {computed,ref,reactive,watch,onBeforeUnmount} from 'vue'
import {api} from '../api'
import OptionalParameters from './OptionalParameters.vue'
import {modelParametersSchema} from '../editor-context'
import {scopeOptions} from '../parameter-overrides'
import {modelParameterSpecs,selectedModelParameters} from '../model-options'
import ModelSelection from './ModelSelection.vue'
import {option,named,pollJob,binningOptions} from '../workflow-parameters'
import WorkflowResults from './WorkflowResults.vue'
import BatchResultPicker from './BatchResultPicker.vue'
import {submitDatasetRuns} from '../dataset-runs'
const props=defineProps({dataset:Object,datasets:Array,runDatasets:{type:Array,required:true},samplePlan:{type:Object,required:true},catalog:Object,compare:Boolean})
const job=ref(null),error=ref(''),busy=ref(false),searchText=ref('{}'),explainJob=ref(null),sampleText=ref('1'),sampleDataset=ref(''),modelFile=ref('model.joblib')
const sampleOverrides=ref({})
const targetConfig=useTargetConfig(()=>props.dataset,'training')
const form=reactive({library_params:{},features:[],models:['logistic'],model_params:{},model_fit_params:{},cv:3,threshold:0.5,search:false,n_iter:10,selector:'none',selector_threshold:0.02,scorecard:false,pdo:20,base_score:600,explain:false})
const columns=computed(()=>props.dataset?.columns||[]),models=computed(()=>props.catalog.models||{}),availableFeatures=computed(()=>columns.value.filter(c=>!targetConfig.sources.includes(c.name)))
sampleDataset.value=props.dataset.id
const selection=modelResultSelection(props.dataset,props.compare)
followModelResult(()=>selection.selected.value.jobId,job,api,message=>{error.value=message})
followModelResult(()=>selection.selected.value.explainId,explainJob,api,message=>{error.value=message})
rememberForm(datasetDraftKey(props.dataset,props.compare?'compare':'experiment'),
  () => ({form,target:targetConfig.draft,searchText:searchText.value,sampleText:sampleText.value,sampleDataset:sampleDataset.value,sampleOverrides:sampleOverrides.value,modelFile:modelFile.value}),
  saved => {
    Object.assign(form,saved.form)
    if (saved.target) copyTargetDraft(targetConfig.draft,saved.target)
    searchText.value=saved.searchText ?? '{}'
    sampleText.value=saved.sampleText ?? '1'
    sampleDataset.value=saved.sampleDataset ?? props.dataset.id
    sampleOverrides.value=saved.sampleOverrides || {}
    modelFile.value=saved.modelFile || 'model.joblib'
  })
watch(()=>targetConfig.sources.slice(),names=>{form.features=form.features.filter(c=>!names.includes(c))})
const searchSchema = computed(() => modelParametersSchema(form.models, models.value))
const specs=computed(()=>[
  option('features','字段选择','选择参与训练的字段，训练和评估标签会自动排除',[],{hint:'优先使用推荐特征，其次使用全部非标签字段',fields:availableFeatures.value,multi:true,limit:200}),
  option('selector','特征选择','在每个训练折内按 IV 或缺失率筛选输入字段','none',{choices:named({none:'不筛选',iv:'IV 信息量筛选',null:'缺失率筛选'})}),
  option('cv','交叉验证折数','按训练集交叉验证 AUC 选择最佳模型',3,{min:2,max:10}),
  option('threshold','分类阈值','用于预测明细、混淆矩阵和发布服务的分类结果',0.5,{min:0.01,max:0.99,step:0.05}),
  option('search','超参数搜索','用 ModelTuner 搜索当前模型的候选值',false),
  option('scorecard','概率转评分','导出的模型同时支持 predict_score',false),
  ...binningOptions('评估报告',props.catalog.parameters?.methods),
  option('date','日期字段','按期查看模型表现',null,{group:'评估报告',fields:columns.value.filter(c=>c.kind==='date')}),
  option('freq','时间频率','报告按此周期统计','M',{group:'评估报告',choices:named({D:'日',W:'周',M:'月',Q:'季度'}),when:d=>!!d.date}),
  option('group','分组字段','按业务维度查看不同群体的表现',null,{group:'评估报告',fields:columns.value}),
  option('amount','金额字段','在评估报告中并列展示订单和金额口径',null,{group:'评估报告',fields:columns.value.filter(c=>c.kind==='numeric')}),
  option('explain','SHAP 与变量交互','解释最多 200 个训练样本，显示变量重要性及依赖图',false,{group:'评估报告'}),
  option('selector_threshold','筛选阈值','IV 下限或允许的缺失率上限',0.02,{min:0,max:1,step:0.01,when:d=>d.selector!=='none'}),
  option('n_iter','搜索次数','限制每个模型的候选参数组合数',10,{min:1,max:100,when:d=>d.search}),
  option('search_spaces','搜索空间','按模型指定候选值，空对象使用推荐空间','{}',{type:'json',schema:searchSchema.value,when:d=>d.search}),
  option('base_score','基准分','概率评分转换的基准分',600,{when:d=>d.scorecard}),
  option('pdo','赔率翻倍分差 PDO','每次赔率翻倍对应的分数变化',20,{min:1,when:d=>d.scorecard}),
  option('random_state','随机种子','用于留出集、交叉验证与模型初始化',42,{group:'训练与验证',min:0,max:2147483647}),
])
const selectedKey=computed(()=>form.models[0])
const selectedModel=computed(()=>models.value[selectedKey.value])
const constructorSpecs=computed(()=>props.compare?[]:modelParameterSpecs(selectedModel.value).map(s=>({...s,key:'model:'+s.key,apiName:s.key})))
const allSpecs=computed(()=>[...specs.value,...constructorSpecs.value])
const settings=computed(()=>({...form,search_spaces:searchText.value,...Object.fromEntries(constructorSpecs.value.map(s=>[s.key,Object.hasOwn(form.model_params[selectedKey.value]||{},s.apiName)?form.model_params[selectedKey.value][s.apiName]:s.default]))}))
const groups=computed(()=>{
  const groups=scopeOptions(props.catalog.parameters,'model')
  if(!props.compare){
    delete groups.comparison
    if(selectedModel.value){
      groups.model={label:'模型构造参数',params:selectedModel.value.extra_params}
      if(Object.keys(selectedModel.value.fit_params||{}).length)groups.fit={label:'模型拟合参数',params:selectedModel.value.fit_params}
    }
  }
  return groups
})
const overrides=computed({
  get(){
    const result={...form.library_params}
    if(!props.compare&&selectedModel.value){
      result.model=Object.fromEntries(Object.entries(form.model_params[selectedKey.value]||{}).filter(([key])=>!Object.hasOwn(selectedModel.value.params,key)))
      result.fit=form.model_fit_params[selectedKey.value]||{}
    }
    return result
  },
  set(values){
    const {model,fit,...library}=values
    form.library_params=library
    if(!props.compare&&selectedModel.value){
      const basic=Object.fromEntries(Object.entries(form.model_params[selectedKey.value]||{}).filter(([key])=>Object.hasOwn(selectedModel.value.params,key)))
      form.model_params[selectedKey.value]={...basic,...model}
      form.model_fit_params[selectedKey.value]=fit||{}
    }
  },
})
function setOption(key,value){if(key==='search_spaces')searchText.value=value;else if(key.startsWith('model:'))form.model_params[selectedKey.value]={...form.model_params[selectedKey.value],[key.slice(6)]:value};else form[key]=value}
let resultRequest=0
async function run(){
  if(props.samplePlan.error)return
  busy.value=true
  error.value=''
  const token=++resultRequest
  const sequence=selection.begin()
  try{
    const body={
      ...form,...targetConfig.payload,...props.samplePlan.payload,kind:'model',workflow:true,
      model_mode:props.compare?'compare':'experiment',
      model_params:props.compare ? selectedModelParameters(form.models,models.value,form.model_params) : Object.fromEntries(form.models.map(k=>[k,form.model_params[k]||{}])),
      model_fit_params:Object.fromEntries(form.models.filter(k=>form.model_fit_params[k]).map(k=>[k,form.model_fit_params[k]])),
      search_spaces:form.search?JSON.parse(searchText.value):{},
    }
    const [created]=await submitDatasetRuns(api,body,props.runDatasets)
    if(token===resultRequest && selection.select(sequence,created.id)){
      job.value=created
      sampleDataset.value=created.dataset_id
      explainJob.value=null
      modelFile.value='model.joblib'
    }
    await refresh()
  }catch(e){error.value=e.message}finally{busy.value=false}
}
async function refresh(){
  try{
    await Promise.all([pollJob(job,api),pollJob(explainJob,api)])
  }catch(e){error.value=e.message}
}
async function open(row){
  const token=++resultRequest
  const sequence=selection.begin()
  try{
    const record=await api('/jobs/'+row.id)
    if(token!==resultRequest || !selection.select(sequence,record.id))return
    job.value=record
    sampleDataset.value=row.dataset_id
    explainJob.value=null
    modelFile.value='model.joblib'
  }catch(e){if(token===resultRequest)error.value=e.message}
}
async function explain(){
  const source=job.value.id
  const sequence=selection.begin()
  try{
    const rows=sampleText.value.split(',').map(v=>Number(v.trim())-1)
    if(rows.some(n=>!Number.isInteger(n)||n<0))throw Error('请输入从 1 开始的行号，多个行号用逗号分隔')
    const created=await api('/jobs',{method:'POST',body:JSON.stringify({kind:'explain',workflow:true,dataset_id:sampleDataset.value,source_job:source,model_file:modelFile.value,sample_rows:rows,library_params:sampleOverrides.value})})
    if(job.value?.id===source && selection.selected.value.jobId===source && selection.select(sequence,source,created.id))explainJob.value=created
  }catch(e){error.value=e.message}
}
async function stop(){try{job.value=await api(`/jobs/${job.value.id}/stop`,{method:'POST'})}catch(e){error.value=e.message}}
const poll=setInterval(refresh,2500);refresh();onBeforeUnmount(()=>clearInterval(poll))
</script>
<template>
  <section class="panel config-panel"><el-alert v-if="error" :title="error" type="error" :closable="false"/>
    <el-form @submit.prevent label-position="top"><TargetConfig :config="targetConfig" :dataset="dataset"/>
      <ModelSelection shared-editor v-model:models="form.models" v-model:parameters="form.model_params" v-model:fit-parameters="form.model_fit_params" :catalog="models" :compare="compare"/>
      <OptionalParameters context="model" :specs="allSpecs" :values="settings" :extensions="groups" v-model:overrides="overrides" @change="setOption"/>

    <div class="submit-bar"><span class="muted">{{runDatasets.length>1?runDatasets.length+' 份训练样本':dataset?.name}} · {{form.models.length}} 个模型</span><el-button type="primary" :disabled="!!targetConfig.error||!!samplePlan.error||!form.models.length" :loading="busy" @click="run">{{compare?'运行模型对比':'运行模型实验'}}</el-button><el-button v-if="job&&['排队中','运行中'].includes(job.status)" @click="stop">停止任务</el-button></div></el-form>
  </section>
  <BatchResultPicker :job="job" @select="open"/>
  <WorkflowResults :job="job"/>
  <section v-if="job?.status==='成功'" class="panel"><h2>样本预测解释</h2><div class="form-grid"><label>模型文件<el-select v-model="modelFile"><el-option v-for="file in job.artifacts.filter(f=>f.startsWith('model')&&f.endsWith('.joblib'))" :key="file" :value="file"/></el-select></label><label>样本数据集<el-select v-model="sampleDataset"><el-option v-for="d in datasets" :key="d.id" :value="d.id" :label="d.name"/></el-select></label><label>样本行号（从1开始，最多10个）<el-input v-model="sampleText" placeholder="1,2,3"/></label><OptionalParameters context="explain" title="解释参数" :specs="[]" :values="{}" :extensions="scopeOptions(catalog.parameters,'explain')" v-model:overrides="sampleOverrides"/><el-button @click="explain">分析样本 SHAP</el-button></div></section>
  <WorkflowResults :job="explainJob"/>
</template>

