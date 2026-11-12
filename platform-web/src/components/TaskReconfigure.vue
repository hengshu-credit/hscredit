<script setup>
import {computed, provide, ref, watch} from 'vue'
import {api} from '../api'
import {taskName} from '../task-center'
import CodeEditor from './CodeEditor.vue'
import {taskConfigSchema} from '../editor-context'
import FieldSelect from './FieldSelect.vue'
import SavedTaskTargets from './SavedTaskTargets.vue'
import {scopeOptions} from '../parameter-overrides'
import {binningOptions} from '../workflow-parameters'
import OptionalParameters from './OptionalParameters.vue'
import ModelSelection from './ModelSelection.vue'
import {selectedModelParameters} from '../model-options'

const props = defineProps({job:Object, datasets:Array, parameterCatalog:Object, catalog:{type:Object,default:()=>({})}})
const emit = defineEmits(['close','created'])
const draft = ref({...JSON.parse(JSON.stringify(props.job.config)),
  dataset_revision:props.job.dataset_revision || props.job.config.dataset_revision || 1})
const advanced = ref(false), json = ref(''), error = ref(''), busy = ref(false)
const columns = ref([]), datasetError = ref('')
const source = ref(null)
provide('fieldPickerDataset',source)
// 编辑完整配置时沿用最近一次可解析的绑定；输入普通参数时不重新加载数据版本或模型目录。
const jsonBinding = ref(null)
watch([advanced, json], ([opened, text]) => {
  if (!opened) { jsonBinding.value = null; return }
  let config
  try { config = JSON.parse(text) } catch { return }
  if (!config || typeof config !== 'object') return
  const binding = {dataset_id: config.dataset_id, dataset_revision: config.dataset_revision, models: Array.isArray(config.models) ? config.models : []}
  if (JSON.stringify(binding) !== JSON.stringify(jsonBinding.value)) jsonBinding.value = binding
})
const binding = computed(() => jsonBinding.value || draft.value)
const configSchema = computed(() => taskConfigSchema(draft.value, columns.value, props.datasets, props.catalog, binding.value.models || []))
function replaceLabels(payload){
  for(const key of ['target','targets','target_order','overdue','dpds','overdue_operator','del_grey','label_mode','positive_label','positive_labels','evaluation_targets','primary_index','handle_missing_target'])delete draft.value[key]
  Object.assign(draft.value,JSON.parse(JSON.stringify(payload)))
}
const parameterContext=computed(()=>{
  const cfg=draft.value
  return cfg.kind==='strategy'?(cfg.workflow?cfg.strategy_mode:'legacy_strategy'):cfg.kind
})
const groups=computed(()=>scopeOptions(props.parameterCatalog,parameterContext.value))
const current = computed(() => props.datasets.find(d => d.id === draft.value.dataset_id))
watch([() => binding.value.dataset_id, () => binding.value.dataset_revision], async ([id,revision], _, onCleanup) => {
  let stale = false
  onCleanup(() => { stale = true })
  columns.value = []
  source.value = null
  datasetError.value = ''
  if (!id || (advanced.value && id !== props.job.dataset_id && !props.datasets.some(d => d.id === id))) return
  try {
    const data = await api('/datasets/' + encodeURIComponent(id) + (revision == null ? '' : '?revision=' + encodeURIComponent(revision)))
    if (!stale) {source.value=data;columns.value=data.columns}
  } catch (e) { if (!stale) datasetError.value = e.message }
}, {immediate:true})
function selectDataset(id) {
  draft.value.dataset_id = id
  draft.value.dataset_revision = current.value?.revision || 1
}
function readConfig() {
  let value
  try { value = JSON.parse(json.value) } catch { throw Error('请检查 JSON 配置格式') }
  if (value?.kind !== props.job.kind) throw Error('请保留原任务类型')
  return value
}
function switchMode() {
  try {
    if (advanced.value) draft.value = readConfig()
    else json.value = JSON.stringify(draft.value, null, 2)
    advanced.value = !advanced.value
    error.value = ''
  } catch (e) { error.value = e.message }
}
async function submit() {
  if (busy.value) return
  busy.value = true
  error.value = ''
  try {
    const config = advanced.value ? readConfig() : {...draft.value}
    if(!advanced.value&&config.kind==='model'){
      if(!config.models?.length)throw Error('请选择对比模型')
      config.model_params=selectedModelParameters(config.models,props.catalog,config.model_params)
      config.model_fit_params=Object.fromEntries(Object.entries(config.model_fit_params||{}).filter(([key])=>config.models.includes(key)))
      config.search_spaces=Object.fromEntries(Object.entries(config.search_spaces||{}).filter(([key])=>config.models.includes(key)))
    }
    if(config.kind==='explore'&&config.workflow&&props.job.config.features?.length){
      if(!config.features?.length)throw Error('请选择探索字段')
      config.feature=config.features[0]
      config.other=null
    }
    const job = await api('/jobs', {method:'POST', body:JSON.stringify(config)})
    emit('created', job)
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
</script>
<template>
  <el-dialog :model-value="true" title="重新配置并执行" width="min(720px, 94vw)" :show-close="!busy" :close-on-click-modal="false" :close-on-press-escape="!busy" @close="emit('close')">
    <div class="panel-heading"><span>{{taskName(job)}} · {{job.dataset_name}}</span><el-button link type="primary" @click="switchMode">{{advanced?'返回常用参数':'编辑完整配置'}}</el-button></div>
    <CodeEditor v-if="advanced" v-model="json" label="完整任务配置" :schema="configSchema"
      :expression-context="job.kind==='ruleset' ? {fields:columns, embedded:true, ruleProperty:'selected_rules'} : undefined" :rows="8"/>
    <el-form v-else label-position="top" class="form-grid">
      <el-form-item label="主样本" required><el-select :model-value="draft.dataset_id" filterable @change="selectDataset">
        <el-option v-if="!current" :value="draft.dataset_id" :label="job.dataset_name+'（原数据集）'"/>
        <el-option v-for="d in datasets" :key="d.id" :value="d.id" :label="d.name"/>
      </el-select></el-form-item>
      <el-form-item label="数据版本" required><el-input-number v-model="draft.dataset_revision" :min="1" :max="current?.revision" :step="1" step-strictly/></el-form-item>
      <SavedTaskTargets v-if="source&&!['interaction','explain'].includes(job.kind)" class="full-width" :config="draft" :dataset="source" @apply="replaceLabels"/>
      <div v-if="job.kind==='model'" class="full-width">
        <ModelSelection v-model:models="draft.models" v-model:parameters="draft.model_params" v-model:fit-parameters="draft.model_fit_params" :catalog="catalog" :compare="draft.model_mode==='compare'||!draft.workflow"/>
      </div>
      <template v-if="job.kind==='explore'&&draft.workflow&&job.config.features?.length">
        <el-form-item label="探索字段（按 X、Y、Z 顺序）" class="full-width"><FieldSelect v-model="draft.features" :fields="columns.filter(c=>c.name!==draft.group)" label="探索字段" multiple :limit="3"/></el-form-item>
        <el-form-item label="分组标签"><FieldSelect v-model="draft.group" :fields="columns.filter(c=>!draft.features.includes(c.name)&&c.unique+(c.missing>0?1:0)<=30)" label="分组标签" clearable/></el-form-item>
      </template>
      <template v-else-if="job.kind==='explore'">
        <el-form-item v-for="(label,key) in {feature:'探索字段',other:'散点对照字段'}" :key="key" :label="label"><FieldSelect v-model="draft[key]" :fields="columns" :label="label" clearable/></el-form-item>
      </template>
      <el-form-item v-else-if="!['ruleset','explain'].includes(job.kind)" label="分析特征" class="full-width"><FieldSelect v-model="draft.features" :fields="columns" label="分析特征" multiple :limit="200"/></el-form-item>
      <el-form-item v-if="['explore','interaction'].includes(job.kind)" label="分布格数"><el-input-number v-model="draft.bins" :min="2" :max="100"/></el-form-item>
      <OptionalParameters v-if="Object.keys(groups).length" class="full-width" title="分析参数" :specs="groups.binning?binningOptions('分箱与规则',parameterCatalog.methods):[]" :values="draft" :extensions="groups" v-model:overrides="draft.library_params" @change="(k,v)=>draft[k]=v"/>
      <el-form-item v-if="job.kind==='model'" label="交叉验证折数"><el-input-number v-model="draft.cv" :min="2" :max="10"/></el-form-item>
      <el-form-item v-if="job.kind==='ruleset'" label="组合逻辑"><el-radio-group v-model="draft.rule_logic"><el-radio-button value="or">任一命中</el-radio-button><el-radio-button value="and">全部命中</el-radio-button></el-radio-group></el-form-item>
      <el-form-item v-if="job.kind==='explain'" label="样本行号（从 1 开始）"><el-select :model-value="draft.sample_rows.map(n=>n+1)" multiple filterable allow-create default-first-option @update:model-value="v=>draft.sample_rows=v.map(n=>Number(n)-1)"/></el-form-item>
    </el-form>
    <el-alert v-if="error||(!advanced&&datasetError)" :title="error||datasetError" type="error" :closable="false"/>
    <template #footer><el-button :disabled="busy" @click="emit('close')">取消</el-button><el-button type="primary" :loading="busy" @click="submit">提交新任务</el-button></template>
  </el-dialog>
</template>
