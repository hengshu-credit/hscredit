<script setup>
import FieldSelect from "./FieldSelect.vue"
import {computed, ref, watch} from 'vue'
import DictionaryTargets from './DictionaryTargets.vue'
import {api} from '../api'

const props = defineProps({config: {type: Object, required: true}, dataset: {type: Object, required: true}, training: {type: Boolean, default: true}})
const draft = computed(() => props.config.draft)
const training = computed(() => props.training && props.config.purpose === 'training')
const preview = ref(null), busy = ref(false), message = ref('')
let generation = 0
watch(() => JSON.stringify([props.dataset.id, props.dataset.revision, props.config.payload]), () => {
  generation++
  preview.value = null
  message.value = ''
  busy.value = false
}, {flush: 'sync'})
async function loadPreview() {
  const current = ++generation
  busy.value = true
  message.value = ''
  try {
    const params = {...props.config.payload}
    delete params.primary_index
    if (typeof params.target === 'string') {
      params.targets = [params.target, ...params.evaluation_targets]
      params.positive_labels = {...params.positive_labels, [params.target]: params.positive_label}
      delete params.target
      delete params.positive_label
      delete params.evaluation_targets
    }
    const result = await api('/datasets/'+props.dataset.id+'/target-preview?revision='+(props.dataset.revision || 1), {method: 'POST', body: JSON.stringify(params)})
    if (current === generation) preview.value = result
  } catch (e) {
    if (current === generation) message.value = e.message
  } finally {
    if (current === generation) busy.value = false
  }
}
</script>

<template>
  <section class="target-editor" aria-label="目标变量配置">
    <h3>标签定义 <small>{{config.purpose==='explore'?'可选':'必选'}} · 选择一种方式</small></h3>
    <el-radio-group v-model="draft.mode" aria-label="标签定义方式">
      <el-radio-button v-if="config.purpose==='explore'" value="none">无标签</el-radio-button>
      <el-radio-button value="target">现有标签</el-radio-button><el-radio-button value="dict">标签字典</el-radio-button><el-radio-button value="overdue">逾期天数</el-radio-button>
    </el-radio-group>
    <p v-if="draft.mode==='none'" class="muted">按全部样本统计字段分布。计算 IV、KS 或 AUC 时请选择标签。</p>
    <template v-else>
      <div class="definition">
        <DictionaryTargets v-if="draft.mode==='dict'" :config="config"/>
        <template v-if="draft.mode==='target'">
          <label>标签字段 <small>必选 · 可多选</small><FieldSelect v-model="draft.fields" :fields="config.binary" multiple :limit="config.purpose==='explore'?30:11" label="标签字段" placeholder="选择二分类字段"/></label>
          <div v-if="draft.fields.length" class="value-list"><div class="value-head"><span>标签字段</span><span>坏样本取值 · 必选</span></div>
            <div v-for="name in draft.fields" :key="name"><span>{{name}}</span><el-select v-model="draft.positives[name]" :aria-label="name+'的坏样本取值'" placeholder="选择一个值"><el-option v-for="v in config.values(name)" :key="v" :label="String(v)" :value="v"/></el-select></div>
          </div>
        </template>
        <div v-if="draft.mode==='overdue'" class="target-grid">
          <label class="span-all">逾期天数字段 <small>必选 · 可多选</small><FieldSelect v-model="draft.overdue" :fields="config.numeric" multiple label="逾期天数字段" placeholder="选择数值字段"/></label>
          <label>坏样本比较符 <small>单选</small><el-select v-model="draft.operator" aria-label="坏样本比较符"><el-option v-for="o in ['>','>=','<','<=']" :key="o" :value="o"/></el-select></label>
          <label>阈值（天）<small>必填 · 可填多个</small><el-input v-model="draft.text" aria-label="逾期阈值" placeholder="例如 1, 7, 30"/></label>
        </div>
      </div>
      <section class="target-section"><h3>样本处理 <small>本次任务的全部标签共用</small></h3>
        <div class="target-grid"><label>缺失标签处理<el-select v-model="draft.missing" aria-label="缺失标签处理"><el-option label="按标签剔除" value="drop"/><el-option label="计为坏样本" value="positive"/><el-option label="计为好样本" value="negative"/><el-option label="发现缺失时报错" value="raise"/></el-select></label>
        <label v-if="draft.mode==='overdue'&&['>','>='].includes(draft.operator)">灰样本处理 <small>可选</small><el-checkbox v-model="draft.grey">剔除 0 &lt; 天数 {{draft.operator==='>'?'≤':'&lt;'}} 阈值的样本</el-checkbox></label>
        <label v-if="draft.mode==='dict'&&draft.entries.some(row=>row.operator==='raw')">灰样本处理 <small>可选</small><el-checkbox v-model="draft.dictionaryGrey">原值标签剔除非 0/1 样本</el-checkbox></label></div>
      </section>
      <section v-if="training && !config.error && config.conditions.length>1" class="target-section"><h3>训练主标签 <small>单选 · 其余口径用于评估</small></h3>
        <el-select v-model="config.primary" filterable aria-label="训练主标签"><el-option v-for="c in config.conditions" :key="c" :value="c"/></el-select></section>
      <p v-if="config.error" class="target-error" role="status">{{config.error}}</p>
      <section v-else class="target-section"><div class="preview-heading"><h3>样本分布 <small>{{config.conditions.length}} 个口径</small></h3>
        <el-button :loading="busy" @click="loadPreview">预览样本分布</el-button></div>
        <p v-if="message" class="target-error" role="alert">{{message}}</p>
        <el-table v-if="preview" :data="preview" stripe max-height="240" size="small">
          <el-table-column prop="标签" label="标签" min-width="120"><template #default="{row,$index}">{{row.标签}} <el-tag v-if="training&&$index===(draft.mode==='target'?0:config.primaryIndex)" size="small">主标签</el-tag></template></el-table-column>
          <el-table-column v-for="k in ['有效样本','好样本','坏样本','剔除样本']" :key="k" :prop="k" :label="k" min-width="72" align="right"/>
          <el-table-column label="坏样本率" min-width="80" align="right"><template #default="{row}">{{row.坏样本率===null?'—':(100*row.坏样本率).toFixed(2)+'%'}}</template></el-table-column>
        </el-table>
      </section>
    </template>
  </section>
</template>
<style scoped>
.target-editor h3,.target-editor p{margin:0 0 10px}.target-editor h3{font-size:14px}.target-editor small{font-weight:400;font-size:12px;color:var(--muted);margin-left:5px}
.definition>label,.target-grid>label{display:block;min-width:0;font-size:13px}
.definition>label>.field-select,.target-grid>label>:is(.el-select,.el-input,.field-select,.el-checkbox){margin-top:5px}
.target-grid>label>.el-select{display:block;width:100%}
.target-grid>label>.el-checkbox{display:flex}
.definition,.target-editor>.muted{margin-top:14px}.target-section{border-top:1px solid var(--line-soft);padding-top:14px;margin-top:14px}.target-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px}.span-all{grid-column:1/-1}
.target-grid>label:only-child{grid-column:1/-1}
.value-list{margin-top:10px}.value-list>div{display:grid;align-items:center;grid-template-columns:minmax(0,1fr) minmax(150px,1fr);gap:10px;padding:5px 10px;border-bottom:1px solid var(--line-soft);overflow-wrap:anywhere}.value-head{color:var(--muted);background:var(--table-head);font-size:12px}
.preview-heading{display:flex;align-items:center;justify-content:space-between}.preview-heading h3{margin:0}.target-error{color:var(--danger)}.target-editor .el-table{margin-top:10px}
@media(max-width:600px){.target-grid{grid-template-columns:1fr}.target-editor .el-radio-group{row-gap:4px}}
</style>
