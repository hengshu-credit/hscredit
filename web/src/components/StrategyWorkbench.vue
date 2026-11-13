<script setup>
import {categoryGroupsSchema} from '../editor-context'
import CodeEditor from './CodeEditor.vue'
import FieldSelect from "./FieldSelect.vue"
import TargetConfig from './TargetConfig.vue'
import {useTargetConfig} from '../target-config'
import {computed,ref,reactive,watch,onBeforeUnmount} from 'vue'
import {ElMessage} from 'element-plus'
import {api} from '../api'
import OptionalParameters from './OptionalParameters.vue'
import {option,choices,named,binningOptions,pollJob} from '../workflow-parameters'
import {scopeOptions} from '../parameter-overrides'
import WorkflowResults from './WorkflowResults.vue'
import BatchResultPicker from './BatchResultPicker.vue'
import {submitDatasetRuns} from '../dataset-runs'
const props=defineProps({catalog:Object,dataset:Object,datasets:Array,runDatasets:{type:Array,required:true},samplePlan:{type:Object,required:true}})
const mode=ref('single'),automatic=ref(true),loading=ref(false),error=ref(''),job=ref(null),selectedRules=ref([]),dragIndex=ref(null)
const emit=defineEmits(['sample-context'])
watch(mode,value=>emit('sample-context',value==='ruleset'?'ruleset':'strategy'),{immediate:true})
const ruleKeys = new WeakMap()
function ruleKey(rule) {
  if (!ruleKeys.has(rule)) ruleKeys.set(rule, Symbol())
  return ruleKeys.get(rule)
}
const basketKey='hscredit-selected-rules:'+props.dataset.id
try{selectedRules.value=JSON.parse(localStorage.getItem(basketKey)||'[]')}catch{}
const section=computed({get:()=>['binary','multiway'].includes(mode.value)?'tree':mode.value,set:value=>{mode.value=value==='tree'?'binary':value}})

const targetConfig=useTargetConfig(()=>props.dataset,()=>mode.value==='ruleset'?'report':'training')
const form=reactive({features:[],library_params:{},method:'quantile',min_n_bins:1,max_n_bins:5,min_bin_size:1,date:null,group:null,amount:null,freq:'M',tree_depth:3,tree_min_samples:10,tree_edits:[],rule_logic:'or',rule_name:'规则集'})
const edit=reactive({node:0,feature:'',values:'',categories:''})
const columns=computed(()=>props.dataset?.columns||[]),features=computed(()=>columns.value.filter(c=>!targetConfig.sources.includes(c.name)))
const categorySchema = computed(() => categoryGroupsSchema(columns.value.find(c => c.name === edit.feature)))
let timer,poll,request=0
watch(()=>props.dataset?.id,()=>{request++;clearTimeout(timer);job.value=null;form.features=[];form.date=null;form.group=null;form.amount=null;form.tree_edits=[]},{immediate:true})
watch(mode,()=>{request++;clearTimeout(timer);loading.value=false;job.value=null;Object.assign(form,{library_params:{},features:[],date:null,group:null,amount:null,method:'quantile',max_n_bins:5,min_n_bins:1,min_bin_size:1,freq:'M',tree_depth:3,tree_min_samples:10,tree_edits:[]})})
watch(()=>targetConfig.sources.slice(),names=>{form.features=form.features.filter(c=>!names.includes(c))})
const specs=computed(()=>[
  option('date','时间字段','按期查看分箱风险和稳定性',null,{fields:columns.value.filter(c=>c.kind==='date')}),
  option('group','分组字段','按渠道等业务分组查看分布',null,{fields:columns.value}),
  option('freq','时间频率','有时间字段时使用的汇总周期','M',{choices:named({D:'日',W:'周',M:'月',Q:'季度'}),when:d=>!!d.date}),
  option('amount','金额字段','在报告中增加金额口径',null,{fields:columns.value.filter(c=>c.kind==='numeric')}),
  ...binningOptions('分箱与规则',props.catalog?.methods),
  ...(mode.value==='binary'?[option('random_state','随机种子','用于训练二叉树',42,{min:0,max:2147483647})]:[]),
  ...(['binary','multiway'].includes(mode.value)?[option('tree_depth','最大树深','限制自动分裂深度',3,{min:1,max:10}),option('tree_min_samples','叶节点最少样本','控制叶节点覆盖量',10,{min:1})]:[]),
])

watch(selectedRules,()=>localStorage.setItem(basketKey,JSON.stringify(selectedRules.value)),{deep:true})
const ready=computed(()=>props.dataset&&!targetConfig.error&&(mode.value==='ruleset'?selectedRules.value.length:mode.value==='single'?form.features.length===1:mode.value==='pair'?form.features.length===2:form.features.length>0))
function config(){return {kind:mode.value==='ruleset'?'ruleset':'strategy',workflow:true,dataset_id:props.dataset.id,...form,...targetConfig.payload,dataset_revision:props.dataset.revision||1,strategy_mode:mode.value==='ruleset'?'single':mode.value,features:mode.value==='ruleset'?[]:form.features,...props.samplePlan.payload,selected_rules:(mode.value==='ruleset'?selectedRules.value:[]).map(({name,expression})=>({name,expression}))}}
const signature=computed(()=>JSON.stringify([props.dataset?.id,mode.value,form,targetConfig.payload,props.samplePlan.payload,props.runDatasets.map(d=>[d.id,d.revision])]))
watch(signature,()=>{
  clearTimeout(timer)
  if(automatic.value&&ready.value&&mode.value!=='ruleset'&&props.runDatasets.length===1)
    timer=setTimeout(()=>{if(automatic.value&&props.runDatasets.length===1)run()},1000)
})
async function run(){
  if(!ready.value)return
  clearTimeout(timer)
  const current=++request
  loading.value=true;error.value=''
  try{
    const [created]=await submitDatasetRuns(api,config(),props.runDatasets)
    if(current===request)job.value=created
    await refresh()
  }catch(e){if(current===request)error.value=e.message}finally{if(current===request)loading.value=false}
}
async function refresh(){try{await pollJob(job,api)}catch(e){error.value=e.message}}
async function open(row){
  clearTimeout(timer)
  const token=++request
  loading.value=false
  try{
    const record=await api('/jobs/'+row.id)
    if(token===request)job.value=record
  }catch(e){if(token===request)error.value=e.message}
}
async function stop(){try{job.value=await api(`/jobs/${job.value.id}/stop`,{method:'POST'})}catch(e){error.value=e.message}}
function addRules(rules){for(const item of rules){let name=item.name,index=2;while(selectedRules.value.some(r=>r.name===name))name=`${item.name} ${index++}`;selectedRules.value.push({name,expression:item.expression})}ElMessage.success('已加入规则集，可在规则集评估页命名、排序与评估')}
function move(from,to){if(from===null||from===to||to<0||to>=selectedRules.value.length)return;const list=[...selectedRules.value];list.splice(to,0,list.splice(from,1)[0]);selectedRules.value=list;dragIndex.value=null}
function addEdit(){try{const base={node:edit.node,feature:edit.feature};if(!base.feature)throw Error('请选择分裂字段');if(mode.value==='binary'){base.threshold=Number(edit.values);if(!edit.values||!Number.isFinite(base.threshold))throw Error('请输入数值阈值')}else if(edit.categories){base.category_groups=JSON.parse(edit.categories)}else{base.thresholds=edit.values.split(',').map(Number);if(!edit.values||base.thresholds.some(v=>!Number.isFinite(v)))throw Error('请输入逗号分隔的数值阈值')}form.tree_edits.push(base)}catch(e){ElMessage.error(e.message)}}
poll=setInterval(refresh,2500);refresh();onBeforeUnmount(()=>{request++;clearTimeout(timer);clearInterval(poll)})
</script>
<template>
  <section class="panel config-panel"><el-tabs v-model="section"><el-tab-pane label="单变量规则挖掘" name="single"/><el-tab-pane label="双变量规则挖掘" name="pair"/><el-tab-pane label="多变量规则挖掘" name="tree"/><el-tab-pane :label="`规则集评估（${selectedRules.length}）`" name="ruleset"/></el-tabs>
    <el-alert v-if="error" :title="error" type="error" :closable="false"/>
    <el-form label-position="top"><TargetConfig :config="targetConfig" :dataset="dataset" :training="mode!=='ruleset'"/><div class="form-grid">
      <el-form-item v-if="section==='tree'" label="树类型" required><el-select v-model="mode"><el-option value="binary" label="手工二叉树"/><el-option value="multiway" label="手工多叉树"/></el-select></el-form-item>
      <el-form-item v-if="mode!=='ruleset'" label="分析字段" required><FieldSelect v-model="form.features" :fields="features.filter(c=>mode!=='binary'||c.kind==='numeric')" multiple :limit="mode==='single'?1:mode==='pair'?2:200" label="分析字段" :placeholder="mode==='single'?'选择 1 个字段':mode==='pair'?'选择 2 个字段':'选择一个或多个字段'"/></el-form-item>
    </div><OptionalParameters :context="mode" :key="mode" :specs="specs" :values="form" :extensions="scopeOptions(catalog,mode)" v-model:overrides="form.library_params" @change="(key,value)=>form[key]=value"/>
</el-form>
    <section v-if="['binary','multiway'].includes(mode)"><el-collapse><el-collapse-item title="手工调整节点分裂"><p class="muted">节点编号来自树规则表。按添加顺序重放分裂，再对所有数据集重新评估。</p><div class="form-grid"><label>节点编号<el-input-number :controls="false" v-model="edit.node" :min="0"/></label><label>分裂字段<FieldSelect v-model="edit.feature" :fields="features.filter(c=>form.features.includes(c.name))" label="分裂字段"/></label><label>阈值（多叉树用逗号分隔）<el-input v-model="edit.values"/></label><div v-if="mode==='multiway'" class="full-width">类别组（JSON，可选）<CodeEditor v-model="edit.categories" :schema="categorySchema" label="类别组" :rows="2" placeholder='[["A"],["B","C"]]'/></div></div><el-button @click="addEdit">添加分裂</el-button><div v-for="(e,i) in form.tree_edits" :key="i" class="rule-row"><span>{{i+1}}. {{e}}</span><el-button @click="form.tree_edits.splice(i,1)">移除</el-button></div></el-collapse-item></el-collapse></section>
    <section v-if="mode==='ruleset'"><div class="form-grid"><label>规则集名称<el-input v-model="form.rule_name"/></label><label>组合逻辑<el-select v-model="form.rule_logic"><el-option label="任一命中（或）" value="or"/><el-option label="全部命中（且）" value="and"/></el-select></label></div><p class="muted">拖拽行调整顺序，也可使用上移／下移。调整后重新评估累积命中与增量效果。</p><div v-for="(rule,i) in selectedRules" :key="ruleKey(rule)" class="rule-row" draggable="true" @dragstart="dragIndex=i" @dragover.prevent @drop.prevent="move(dragIndex,i)"><span>{{i+1}}</span><el-input v-model="rule.name" aria-label="规则命名"/><CodeEditor v-model="rule.expression" :expression-context="{fields:features}" label="规则表达式" language="expression" :rows="1"/><el-button :disabled="i===0" @click="move(i,i-1)">上移</el-button><el-button :disabled="i===selectedRules.length-1" @click="move(i,i+1)">下移</el-button><el-button @click="selectedRules.splice(i,1)">移除</el-button></div><el-button @click="selectedRules.push({name:'手写规则 '+(selectedRules.length+1),expression:''})">添加手写规则</el-button></section>
    <div class="submit-bar"><el-checkbox v-if="mode!=='ruleset'&&runDatasets.length===1" v-model="automatic">配置完成后自动生成</el-checkbox><el-button :disabled="!ready" :loading="loading" type="primary" @click="run">{{mode==='ruleset'?'评估规则集并生成置换报告':'生成分析'}}</el-button><el-button v-if="job&&['排队中','运行中'].includes(job.status)" @click="stop">停止任务</el-button></div>
  </section>
  <BatchResultPicker :job="job" @select="open"/>
  <WorkflowResults :job="job" :selectable="mode!=='ruleset'" @select="addRules"/>

</template>
<style scoped>
.rule-row{display:flex;gap:8px;align-items:center;margin:10px 0;padding:10px;border:1px solid var(--border);background:var(--surface)}.rule-row .el-input{min-width:120px}.rule-row .el-input{flex:1}.rule-row .code-editor{flex:2;min-width:160px}.form-grid label{display:grid;gap:8px}.workflow-results{margin-top:16px}@media(max-width:900px){.rule-row{flex-wrap:wrap}.rule-row .el-input,.rule-row .code-editor{flex-basis:100%}}
</style>
