<script setup>
import CodeEditor from './CodeEditor.vue'
import FieldSelect from "./FieldSelect.vue"
import {computed,nextTick,provide,ref,watch} from 'vue'
import {ElMessage,ElMessageBox} from 'element-plus'
import {api} from '../api'
import {emptyNode,fieldNode,literalNode,operationNode,replaceNode,fieldsBefore,describeNode,configuredRules,expressionComplete,switchRuleMode} from '../processing'
import FxNode from './FxNode.vue'
import FieldContentPicker from './FieldContentPicker.vue'
import DataTable from './DataTable.vue'
const props=defineProps({modelValue:Boolean,dataset:Object,catalog:Object})
provide('fieldPickerDataset',computed(()=>props.dataset))
const emit=defineEmits(['update:modelValue','saved'])
const rules=ref([]),active=ref(0),error=ref(''),preview=ref(null),previewing=ref(false),saving=ref(false),saveAs=ref(false),name=ref(''),tab=ref('editor')
const picker=ref(false),pickPath=ref([]),previewKey=ref('');let requestNumber=0
const formulaEditing=ref(false),formulaText=ref(''),formulaError=ref(''),formulaParsing=ref(false),formulaInput=ref()
const drafts=new Map(),textDrafts=new Map();let loadedDataset=''
const fields=computed(()=>fieldsBefore(props.dataset?.columns||[],rules.value,active.value))
const current=computed(()=>rules.value[active.value]),revision=computed(()=>props.dataset?.revision||1)
const fingerprint=computed(()=>JSON.stringify({id:props.dataset?.id,revision:revision.value,rules:configuredRules(rules.value)}))
const rulesValid=computed(()=>rules.value.length>0&&rules.value.every(r=>r.target.trim()&&expressionComplete(r.expression)))
const canApply=computed(()=>preview.value&&!formulaEditing.value&&!formulaParsing.value&&previewKey.value===fingerprint.value&&(!saveAs.value||name.value.trim()))
const history=computed(()=>props.dataset?.history?.length?props.dataset.history:[{revision:revision.value,at:props.dataset?.created_at,description:'初始数据',columns:props.dataset?.columns?.length}])
watch(fingerprint,()=>{preview.value=null;previewKey.value='';requestNumber++})
watch([()=>props.modelValue,()=>props.dataset?.id],([opened,id])=>{
  if(!opened){if(loadedDataset)drafts.set(loadedDataset,{rules:structuredClone(configuredDraft()),active:active.value,saveAs:saveAs.value,name:name.value});return}
  error.value='';preview.value=null;previewKey.value='';tab.value='editor'
  if(id===loadedDataset&&rules.value.length)return
  loadedDataset=id
  const saved=drafts.get(id)
  rules.value=saved?.rules||[];active.value=saved?.active||0;saveAs.value=props.dataset?.storage==='indices'||saved?.saveAs||false;name.value=saved?.name||`${props.dataset?.name||'数据'} · 加工结果`
  if(!saved)addRule()
})
function configuredDraft(){return JSON.parse(JSON.stringify(rules.value))}
watch(()=>current.value?._id,(id,previous)=>{
  if(previous&&formulaEditing.value)textDrafts.set(previous,formulaText.value)
  formulaEditing.value=!!id&&textDrafts.has(id);formulaText.value=textDrafts.get(id)||'';formulaError.value=''
})
async function startFormulaEdit(){if(!current.value||saving.value)return;formulaText.value=textDrafts.get(current.value._id)??describeNode(current.value.expression);formulaEditing.value=true;formulaError.value='';preview.value=null;await nextTick();formulaInput.value?.focus()}
function cancelFormulaEdit(){if(current.value)textDrafts.delete(current.value._id);formulaEditing.value=false;formulaError.value=''}
async function parseFormula(){
  if(formulaParsing.value||!current.value||!formulaText.value.trim())return
  const id=current.value._id,key=fingerprint.value,text=formulaText.value
  formulaParsing.value=true;formulaError.value=''
  try{
    const value=await api(`/datasets/${props.dataset.id}/processing/parse`,{method:'POST',body:JSON.stringify({text,expected_revision:revision.value,prior_rules:configuredRules(rules.value.slice(0,active.value).filter(r=>r.target.trim()&&expressionComplete(r.expression)))})})
    if(current.value?._id!==id||fingerprint.value!==key||formulaText.value!==text){formulaError.value='配置已变化，请重新解析';return}
    current.value.expression=value.expression;textDrafts.delete(id);formulaEditing.value=false
  }catch(e){formulaError.value=e.message}finally{formulaParsing.value=false}
}
function addRule(){const first=fields.value.find(f=>f.valueType==='numeric')?.name||fields.value[0]?.name;rules.value.push({_id:crypto.randomUUID(),mode:'add',target:'',expression:operationNode(fieldNode(first))});active.value=rules.value.length-1}
function removeRule(index){rules.value.splice(index,1);active.value=Math.max(0,Math.min(active.value,rules.value.length-1))}
function moveRule(direction){const to=active.value+direction;if(to<0||to>=rules.value.length)return;[rules.value[to],rules.value[active.value]]=[rules.value[active.value],rules.value[to]];active.value=to}
function chooseField(field){current.value.expression=replaceNode(current.value.expression,pickPath.value,fieldNode(field))}
function setMode(mode){Object.assign(current.value,switchRuleMode(current.value,mode,fields.value))}
function template(kind){cancelFormulaEdit();const value=fields.value.find(f=>f.valueType==='numeric')?.name||'',f=()=>fieldNode(value),n=literalNode,op=(a,o,b)=>({kind:'OPERATION',terms:[{operand:a},{operator:o,operand:b}]});if(kind==='scale')current.value.expression=op(f(),'*',n(1));if(kind==='missing')current.value.expression={kind:'FUNCTION',name:'where',args:[{kind:'FUNCTION',name:'isnan',args:[f()]},n(0),f()]};if(kind==='clip')current.value.expression={kind:'FUNCTION',name:'clip',args:[f(),n(0),n(100)]};if(kind==='label')current.value.expression={kind:'FUNCTION',name:'where',args:[op(f(),'>',n(30)),n(1),n(0)]}}
async function doPreview(){const key=fingerprint.value,number=++requestNumber;previewing.value=true;error.value='';preview.value=null;try{const result=await api(`/datasets/${props.dataset.id}/processing/preview`,{method:'POST',body:JSON.stringify({rules:configuredRules(rules.value),expected_revision:revision.value})});if(number===requestNumber&&key===fingerprint.value){preview.value=result;previewKey.value=key}}catch(e){if(number===requestNumber)error.value=e.message}finally{previewing.value=false}}
async function apply(){if(!canApply.value)return;saving.value=true;error.value='';try{const result=await api(`/datasets/${props.dataset.id}/processing/apply`,{method:'POST',body:JSON.stringify({rules:configuredRules(rules.value),expected_revision:preview.value.revision,save_as:saveAs.value,name:saveAs.value?name.value:null})});ElMessage.success(saveAs.value?'已另存加工后的数据集':'字段加工已保存');emit('saved',result);emit('update:modelValue',false)}catch(e){error.value=e.message;preview.value=null}finally{saving.value=false}}
async function restore(version){try{await ElMessageBox.confirm(`将当前数据恢复为版本 ${version} 的内容，并保留全部历史版本。`,'恢复数据版本',{confirmButtonText:'恢复',cancelButtonText:'取消',type:'warning'})}catch{return}saving.value=true;try{const result=await api(`/datasets/${props.dataset.id}/processing/restore`,{method:'POST',body:JSON.stringify({revision:version,expected_revision:revision.value})});emit('saved',result);emit('update:modelValue',false);ElMessage.success('已恢复数据版本')}catch(e){error.value=e.message}finally{saving.value=false}}
</script>
<template><el-dialog :model-value="modelValue" title="字段加工" width="min(1320px, 96vw)" top="3vh" :close-on-click-modal="false" :close-on-press-escape="!saving" :show-close="!saving" class="processing-dialog" @update:model-value="emit('update:modelValue',$event)">
  <div class="processing-dataset"><strong>{{dataset?.name}}</strong><el-tag effect="plain">版本 {{revision}}</el-tag><span>{{dataset?.rows.toLocaleString()}} 行 · {{dataset?.columns.length}} 个字段</span></div>
  <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" class="error-alert"/>
  <el-tabs v-model="tab"><el-tab-pane label="ƒx 加工配置" name="editor"><div class="processing-layout">
    <aside class="processing-steps"><div class="processing-section-label">配置字段 <small>按顺序执行</small></div><button v-for="(rule,index) in rules" :key="rule._id" :class="{active:active===index}" @click="active=index"><span class="processing-step-number">{{index+1}}</span><div><strong>{{rule.target||'未命名字段'}}</strong><small>{{rule.mode==='add'?'新增字段':'更新字段'}}</small></div></button><el-button :disabled="rules.length>=30||saving||formulaParsing" @click="addRule">＋ 添加字段</el-button></aside>
    <div v-if="current" class="processing-editor"><div class="processing-rule-toolbar"><h3>字段 {{active+1}} · {{current.mode==='add'?'新增字段':'更新原字段'}}</h3><div><el-button size="small" :disabled="active===0||saving" @click="moveRule(-1)">上移</el-button><el-button size="small" :disabled="active===rules.length-1||saving" @click="moveRule(1)">下移</el-button><el-button size="small" type="danger" :disabled="saving||formulaParsing" @click="removeRule(active)">删除字段</el-button></div></div>
      <el-form label-position="top" :disabled="saving"><div class="form-grid"><el-form-item label="处理方式"><el-radio-group :model-value="current.mode" @update:model-value="setMode"><el-radio-button value="add">新增字段</el-radio-button><el-radio-button value="replace">更新原字段</el-radio-button></el-radio-group></el-form-item><el-form-item label="目标字段" class="span-two" required><el-input v-if="current.mode==='add'" v-model="current.target" placeholder="输入新字段名称" aria-label="新字段名称" maxlength="150"/><FieldSelect v-else v-model="current.target" :fields="fields" label="待更新字段"/></el-form-item></div>
      <div class="fx-toolbar"><strong>ƒx 组合表达式</strong><el-dropdown @command="template"><el-button size="small">常用加工 ⌄</el-button><template #dropdown><el-dropdown-menu><el-dropdown-item command="scale">乘系数 / 单位换算</el-dropdown-item><el-dropdown-item command="missing">数值缺失填补</el-dropdown-item><el-dropdown-item command="clip">上下限截断</el-dropdown-item><el-dropdown-item command="label">条件标签</el-dropdown-item></el-dropdown-menu></template></el-dropdown><el-button size="small" @click="current.expression=operationNode(current.expression)">包入一层括号</el-button></div>
      <p class="muted">选择字段或常量，追加运算符；函数参数和运算项都支持继续组合。可引用前序字段的结果。</p>
      <FxNode :key="current._id" v-model="current.expression" :fields="fields" :catalog="catalog" @pick="pickPath=$event;picker=true" @remove="current.expression=emptyNode()"/>
      <div class="fx-formula-preview" :class="{'is-editing':formulaEditing}">
        <div class="fx-preview-heading"><span>组合预览</span><el-button v-if="!formulaEditing" text @click="startFormulaEdit">编辑表达式</el-button><span v-else class="muted">支持字段名、col("字段名") 和面板中的函数</span></div>
        <template v-if="formulaEditing"><CodeEditor ref="formulaInput" v-model="formulaText" language="expression" :expression-context="{fields, functions:catalog.functions, processing:true}" :rows="3" :disabled="saving" label="编辑组合表达式" placeholder='例如：where(col("MOB6") > 30, 1, 0)' @keydown.capture.ctrl.enter.stop.prevent="parseFormula" @keydown.capture.meta.enter.stop.prevent="parseFormula"/><el-alert v-if="formulaError" :title="formulaError" type="error" :closable="false"/><div class="fx-text-actions"><el-button :disabled="formulaParsing" @click="cancelFormulaEdit">取消编辑</el-button><el-button type="primary" :disabled="!formulaText.trim()" :loading="formulaParsing" @click="parseFormula">解析并回显</el-button></div></template>
        <code v-else role="button" tabindex="0" aria-label="双击编辑组合表达式" @dblclick="startFormulaEdit" @keydown.enter.prevent="startFormulaEdit">{{describeNode(current.expression)||'双击输入表达式，或在上方选择内容'}}</code>
      </div>
      </el-form>
    </div><el-empty v-else description="添加字段开始配置"/>
  </div>
  <div class="processing-preview-toolbar"><span class="muted">预览使用全部数据计算，展示前 20 行；保存前可检查字段类型、缺失和无穷值。</span><el-button type="primary" :disabled="!rulesValid||saving||formulaEditing||formulaParsing" :loading="previewing" @click="doPreview">计算预览</el-button></div>
  <section v-if="preview" class="processing-preview"><div class="panel-heading"><h3>加工结果预览</h3><el-tag type="success">已计算 {{preview.rows.toLocaleString()}} 行</el-tag></div><DataTable :rows="preview.statistics" height="240"/><el-alert v-if="preview.statistics.some(s=>s['无穷值数'])" title="结果包含无穷值，可用条件取值或 isfinite 函数处理后再保存。" type="warning" :closable="false"/><div class="processing-compare"><section><h3>处理前</h3><DataTable :rows="preview.before" height="300"/></section><section><h3>处理后</h3><DataTable :rows="preview.after" height="300"/></section></div><el-collapse><el-collapse-item title="生成的 NumExprDerive 表达式"><div v-for="(step,index) in preview.plan" :key="index" class="fx-generated"><strong>{{index+1}}. {{step.target}}</strong><code>{{step.expression}}</code></div></el-collapse-item></el-collapse></section>
  </el-tab-pane><el-tab-pane label="加工记录与版本" name="history"><p class="muted">恢复会生成新版本；已完成的训练和报告继续绑定其原始数据版本。</p><el-table :data="history" stripe><el-table-column prop="revision" label="版本" width="76"/><el-table-column prop="description" label="处理内容" min-width="208"/><el-table-column prop="columns" label="字段数" width="84"/><el-table-column label="时间" min-width="164"><template #default="{row}">{{new Date(row.at).toLocaleString()}}</template></el-table-column><el-table-column label="操作" width="100"><template #default="{row}"><el-tag v-if="row.revision===revision" type="success">当前版本</el-tag><el-button v-else link type="primary" :disabled="saving" @click="restore(row.revision)">恢复此版本</el-button></template></el-table-column></el-table><section v-if="dataset?.processing?.plan" class="processing-history-config"><h3>最近一次加工配置</h3><div v-for="(step,index) in dataset.processing.plan" :key="index" class="fx-generated"><strong>{{step.target}}</strong><code>{{step.expression}}</code></div></section></el-tab-pane></el-tabs>
  <template #footer><div class="processing-footer"><div v-if="tab==='editor'"><el-radio-group v-model="saveAs" :disabled="saving"><el-radio :value="false" :disabled="dataset?.storage==='indices'">更新当前数据集</el-radio><el-radio :value="true">另存为新数据集</el-radio></el-radio-group><el-input v-if="saveAs" v-model="name" aria-label="另存数据集名称" placeholder="新数据集名称" maxlength="150" style="width:260px;margin-left:10px"/></div><div><el-button :disabled="saving" @click="emit('update:modelValue',false)">关闭</el-button><el-button v-if="tab==='editor'" type="primary" :disabled="!canApply||previewing" :loading="saving" @click="apply">{{saveAs?'保存为新数据集':'应用加工并保存'}}</el-button></div></div></template>
  <FieldContentPicker v-model="picker" :fields="fields" :selected="pickPath.reduce((node,key)=>node?.[key],current?.expression)?.name" confirm-label="插入字段" @select="chooseField"/>
</el-dialog></template>
