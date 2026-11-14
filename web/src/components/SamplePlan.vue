<script setup>
import {computed,ref,watch} from 'vue'
import {sampleUses} from '../sample-plan'
import SamplePicker from './SamplePicker.vue'
const props=defineProps({context:String, primary:String, tree:Array, plan:Object, compact:Boolean})
const emit=defineEmits(['primary'])
const use=computed(()=>sampleUses[props.context])
const opened=ref(false), d=ref(null)
const runs=computed(()=>[props.primary,...(d.value?.batch||[])])
function edit(){d.value=JSON.parse(JSON.stringify(props.plan.draft));opened.value=true}
const error=computed(()=>{
  if(!d.value||props.context!=='model')return ''
  if(d.value.testMode==='external'&&!d.value.test)return '请选择独立测试样本'
  if(d.value.testMode==='holdout'&&!(d.value.fraction>=0.1&&d.value.fraction<=0.5))return '留出比例须在 10% 至 50% 之间'
  return ''
})
function apply(){if(!error.value){Object.assign(props.plan.draft,d.value);opened.value=false}}
watch([()=>props.context,()=>props.primary,()=>props.plan.runDatasets[0]?.revision],()=>{opened.value=false})
const summary=computed(()=>{
  const saved=props.plan.draft, parts=[]
  if(props.context==='model')parts.push(({external:'独立测试集',holdout:'分层留出 '+Math.round(saved.fraction*100)+'%',train:'训练集内评估'})[saved.testMode])
  if(saved.evaluations.length)parts.push(saved.evaluations.length+' 份额外评估样本')
  if(saved.batch.length)parts.push('批量 '+(saved.batch.length+1)+' 份')
  if(saved.psi)parts.push('已选 PSI 对照样本')
  return parts.join(' · ')
})
const testId=computed(()=>props.context==='model'&&d.value?.testMode==='external'?d.value.test:'')
const batchExcluded=computed(()=>[props.primary,...d.value.evaluations,testId.value,d.value.psi].filter(Boolean))
const percent=computed({get:()=>d.value.fraction == null ? null : Math.round(d.value.fraction*100),set:v=>{d.value.fraction=v==null?null:v/100}})
</script>
<template>
  <section v-if="use" class="sample-plan panel" aria-label="数据集配置">
    <div class="sample-toolbar"><div class="primary-picker"><label>{{use.primary}} <small>必选</small></label>
      <SamplePicker :model-value="primary" :tree="tree" :label="use.primary" placeholder="选择原表或划分子集" @update:model-value="emit('primary',$event)"/>
      <small v-if="plan.runDatasets[0]">{{plan.runDatasets[0].rows.toLocaleString()}} 行 · 版本 {{plan.runDatasets[0].revision||1}}</small>
    </div><div class="sample-links"><el-button v-if="primary" plain @click="edit">配置样本</el-button><slot/></div></div>
    <p v-if="summary" class="sample-help">{{summary}}</p>
    <p v-if="plan.sharedRows.length" class="sample-overlap" role="status">{{plan.sharedRows.join('、')}}有父子包含关系，拟合和评估选择的样本存在重叠。</p>
    <el-dialog v-model="opened" title="配置样本" width="min(680px, 95vw)" append-to-body destroy-on-close :close-on-click-modal="false">
      <div v-if="d" class="parameter-body"><p class="muted">{{use.help}}</p><div class="sample-grid">
        <div v-if="context==='model'" class="sample-field">
          <label>主测试方式 <small>三选一</small></label>
          <el-select v-model="d.testMode" aria-label="主测试方式">
            <el-option value="holdout" label="分层随机留出"/><el-option value="external" label="独立测试集"/><el-option value="train" label="训练集内评估"/>
          </el-select>
          <div v-if="d.testMode==='holdout'" class="sample-percent"><span>留出</span><el-input-number v-model="percent" aria-label="留出比例（百分比）" :min="10" :max="50" :step="5" :controls="false"/><span>% 用于测试</span></div>
          <SamplePicker v-if="d.testMode==='external'" v-model="d.test" :tree="tree" label="独立测试样本" placeholder="请选择独立测试样本（必选）" :excluded="[...runs,...d.evaluations]"/>
          <small v-if="d.testMode==='train'">主测试指标取自训练样本。</small>
        </div>
        <div v-if="['model','strategy','ruleset'].includes(context)" class="sample-field">
          <label>其他评估样本 <small>可选 · 可多选</small></label>
          <SamplePicker v-model="d.evaluations" :tree="tree" multiple :limit="20" :excluded="[...runs,testId]" label="其他评估样本" placeholder="选择验证集、OOT 或其他样本"/>
        </div>
        <div v-if="context==='summary'" class="sample-field">
          <label>PSI 对照样本 <small>可选</small></label>
          <SamplePicker v-model="d.psi" :tree="tree" :excluded="runs" label="PSI 对照样本" placeholder="默认在分析样本内随机拆分"/>
          <small v-if="d.psi">按同名特征计算两份样本间的 PSI，对照样本无需标签。</small>
        </div>
        <div class="sample-field"><label>追加独立运行样本 <small>可选 · 可多选</small></label>
          <SamplePicker v-model="d.batch" :tree="tree" :excluded="batchExcluded" multiple :limit="9" label="追加独立运行样本"/>
          <small>沿用主样本的字段和配置，每份追加样本单独执行并保留结果。</small>
        </div>
        </div><p v-if="error" class="sample-error" role="alert">{{error}}</p></div>
      <template #footer><el-button @click="opened=false">取消</el-button><el-button type="primary" :disabled="!!error" @click="apply">应用配置</el-button></template>
    </el-dialog>
  </section>
</template>
<style scoped>
.sample-toolbar,.sample-links,.sample-percent{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.sample-plan h2,.sample-plan p{margin:0}
.sample-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(210px,100%),1fr));align-items:start;gap:12px 20px}
.sample-field{display:grid;gap:6px;min-width:0}
.sample-field label{font-weight:650;font-size:14px}
.sample-field small,.sample-help,.sample-percent{font-size:12px;color:var(--muted)}
.sample-field label small{margin-left:4px;font-weight:400}
.sample-percent .el-input-number{width:72px}
.sample-plan .sample-help{margin:10px 0}
.sample-plan .sample-error{color:var(--danger);margin-top:8px}
.sample-plan .sample-overlap{color:var(--warning);font-size:13px;margin:8px 0}
.sample-toolbar,.primary-picker{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.sample-toolbar{justify-content:space-between}.primary-picker>label{font-size:14px;font-weight:600}.primary-picker small{font-size:12px;color:var(--muted)}.primary-picker{flex:1 1 320px;min-width:0}.primary-picker :deep(.el-select){width:min(260px,100%)}.sample-plan .sample-help{margin:6px 0 0}
</style>
