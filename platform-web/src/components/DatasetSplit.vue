<script setup>
import {computed,provide,ref,watch} from 'vue'
import {ElMessage} from 'element-plus'
import {api} from '../api'
import DataTable from './DataTable.vue'
import SplitEditor from './SplitEditor.vue'
import {datasetPlans,uniqueSampleName,restoreSplitConfig,sampleRows} from '../split-editor'
import {datasetNode,serializeTree,treeError,leafNames} from '../split-tree'
import '../split-tree.css'
import '../split-editor.css'

const props=defineProps({modelValue:Boolean,dataset:Object,planId:String,datasets:{type:Array,default:()=>[]}})
provide('fieldPickerDataset',computed(()=>props.dataset))
const emit=defineEmits(['update:modelValue','saved'])
const preview=ref(null),previewKey=ref(''),error=ref(''),previewing=ref(false),saving=ref(false)
const tree=ref(datasetNode('')),editorKey=ref(0)
const plan=ref(null)
const groupPath=computed(()=>props.dataset?.group_path)
const plans=computed(()=>datasetPlans(props.datasets.filter(d=>!d.group_path),groupPath.value?props.dataset.parent_id:props.dataset?.id))
const siblingNames=ref([])
const occupied=computed(()=>[...siblingNames.value,...props.datasets.filter(d=>!d.group_path&&(!plan.value||d.splitting?.plan_id!==plan.value.plan_id)).map(d=>d.name)])
const query=computed(()=>plan.value?'?'+new URLSearchParams({plan_id:plan.value.plan_id,plan_revision:plan.value.plan_revision}):'')
function selectPlan(id){
  const members=plans.value.get(id)||[]
  plan.value=members[0]?.splitting||null
  siblingNames.value=[]
  tree.value=plan.value?restoreSplitConfig(plan.value.config,props.dataset,members):datasetNode(uniqueSampleName(`${props.dataset.name}_划分结果`,occupied.value))
  if(groupPath.value){
    const branch=sampleRows(tree.value).find(row=>row.key===JSON.stringify(groupPath.value))
    if(!branch)throw new Error('分组已更新，请刷新后重新打开')
    const owned=new Set(leafNames(branch.node))
    siblingNames.value=leafNames(tree.value).filter(name=>!owned.has(name))
    tree.value=branch.node
  }
  editorKey.value++
  requestNumber++
  previewing.value=false
  preview.value=null
  error.value=''
}
const outputCount=computed(()=>tree.value.type==='dataset'?0:preview.value?preview.value.parts.filter(p=>p.rows>0).length:leafNames(tree.value).length)
let requestNumber=0
const revision=computed(()=>props.dataset?.revision||1),columns=computed(()=>props.dataset?.columns||[])
const payload=computed(()=>({expected_revision:revision.value,tree:serializeTree(tree.value)})),fingerprint=computed(()=>JSON.stringify([props.dataset?.id,payload.value]))
const validation=computed(()=>tree.value.type==='dataset'?'请选择一种方式开始划分':treeError(tree.value,occupied.value))
const canSave=computed(()=>preview.value&&previewKey.value===fingerprint.value&&!validation.value&&!previewing.value&&!saving.value)
watch(()=>props.modelValue,open=>{
  if(open)selectPlan(props.planId||plans.value.keys().next().value)
  else requestNumber++
})
watch(fingerprint,()=>{preview.value=null;error.value=''})
async function doPreview(){
  if(validation.value)return
  const number=++requestNumber,key=fingerprint.value,id=props.dataset.id,body=JSON.stringify(payload.value)
  previewing.value=true;preview.value=null;error.value=''
  try{const result=await api(`/datasets/${id}/split-tree/preview${query.value}`,{method:'POST',body});if(number===requestNumber&&key===fingerprint.value&&props.modelValue){preview.value=result;previewKey.value=key}}
  catch(e){if(number===requestNumber&&key===fingerprint.value)error.value=e.message}
  finally{if(number===requestNumber)previewing.value=false}
}
async function apply(){
  if(!canSave.value)return
  saving.value=true;error.value=''
  try{const result=await api(`/datasets/${props.dataset.id}/split-tree/apply${query.value}`,{method:'POST',body:JSON.stringify(payload.value)});emit('saved',result.datasets);emit('update:modelValue',false);ElMessage.success(`已保存划分方案，共 ${result.datasets.length} 个子集`)}
  catch(e){error.value=e.message;preview.value=null}
  finally{saving.value=false}
}
</script>

<template>
  <el-dialog :model-value="modelValue" title="划分数据集" width="min(1320px, 97vw)" top="4vh" :close-on-click-modal="false" :close-on-press-escape="!saving" :show-close="!saving" @update:model-value="emit('update:modelValue',$event)">
    <div class="split-source"><strong>{{dataset?.name}}</strong><el-tag effect="plain">版本 {{revision}}</el-tag><span class="muted">{{dataset?.rows.toLocaleString()}} 行 · {{columns.length}} 个字段</span></div>
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false"/>
    <el-form label-position="top" :disabled="saving">
      <el-form-item v-if="plans.size&&!groupPath" label="划分方案"><el-select :model-value="plan?.plan_id||'new'" aria-label="划分方案" @change="selectPlan">
        <el-option label="新建划分方案" value="new"/>
        <el-option v-for="[id,members] in plans" :key="id" :value="id" :label="members.map(d=>d.name).join('、')"/>
      </el-select></el-form-item>
      <SplitEditor :key="editorKey" :node="tree" :dataset="dataset" :existing="occupied" :disabled="saving" @replace="tree=$event"/>
      <div class="split-tree-preview-action"><span class="muted" role="status">{{validation||'预览各子集的样本后保存划分'}}</span><el-button type="primary" :disabled="!!validation||saving" :loading="previewing" @click="doPreview">预览划分结果</el-button></div>
    </el-form>
    <section v-if="preview" class="split-results">
      <h3>{{groupPath?'整体方案划分结果预览':'划分结果预览'}}</h3>
      <el-alert v-if="preview.excluded_rows" :title="`本次有 ${preview.excluded_rows.toLocaleString()} 行被排除，不进入结果数据集；可展开查看划分明细。`" type="warning" :closable="false" show-icon/>
      <el-alert v-if="preview.empty_datasets?.length" class="split-tree-empty" :title="`以下结果没有样本，将不创建空数据集：${preview.empty_datasets.join('、')}`" type="info" :closable="false"/>
      <el-table :data="preview.parts" stripe><el-table-column prop="name" label="数据集名称" min-width="156"/><el-table-column prop="rows" label="实际行数" width="92"/><el-table-column label="占源数据比例" width="108"><template #default="{row}">{{(row.fraction*100).toFixed(2)}}%</template></el-table-column><el-table-column v-if="preview.parts.some(p=>p.groups!==undefined)" prop="groups" label="分组数" width="84"/><el-table-column v-if="preview.parts.some(p=>p.time_min)" label="时间范围" min-width="220"><template #default="{row}">{{row.time_min?`${row.time_min} ～ ${row.time_max}`:"—"}}</template></el-table-column><el-table-column prop="path" label="划分路径" min-width="250" show-overflow-tooltip/></el-table>
      <el-collapse v-if="preview.trace"><el-collapse-item title="查看逐步划分明细"><DataTable :rows="preview.trace.map(t=>({'路径':t.path,'动作':t.action,'输入行数':t.input_rows??null,'命中或输出行数':t.rows,'数据集':t.name||'','排除原因':t.reason||''}))" height="360"/></el-collapse-item></el-collapse>
      <el-collapse><el-collapse-item v-for="(part,index) in preview.parts" :key="index" :title="`${part.name} · 前 5 行样本`" :name="index"><DataTable :rows="part.preview" height="260"/><template v-if="part.strata"><h4>分层类别样本数（最多显示 50 类）</h4><DataTable :rows="part.strata" height="200"/></template></el-collapse-item></el-collapse>
    </section>
    <template #footer><div class="split-footer"><span v-if="groupPath" class="muted">正在编辑「{{dataset.name}}」分支，预览并保存后会更新「{{plan?.source_name}}」的整体划分。</span><span v-else class="muted">子集按行索引引用「{{dataset?.name}}」的版本 {{revision}}。可在源数据集下重新调整划分。</span><div><el-button :disabled="saving" @click="emit('update:modelValue',false)">取消</el-button><el-button type="primary" :disabled="!canSave" :loading="saving" @click="apply"><template v-if="groupPath">更新整体方案</template><template v-else>{{plan?'更新划分':'保存划分'}}（{{outputCount}} 个子集）</template></el-button></div></div></template>
  </el-dialog>
</template>

<style scoped>
.split-source,.split-footer{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.split-source{margin-bottom:20px}.split-results{margin-top:24px}.split-results .el-alert{margin-bottom:12px}.split-footer{justify-content:space-between}.split-footer>div{display:flex;gap:8px}.split-footer .el-button+.el-button{margin-left:0}@media(max-width:650px){.split-footer>div{margin-left:auto}}
</style>

