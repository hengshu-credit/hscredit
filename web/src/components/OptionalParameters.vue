<script setup>
import {computed,ref,watch} from 'vue'
import {Search,Setting} from '@element-plus/icons-vue'
import FieldSelect from './FieldSelect.vue'
import CodeEditor from './CodeEditor.vue'
import {availableParameters,parameterRows,parseRows} from '../parameter-overrides'
import {orderParameters,parameterGroup} from '../workflow-parameters'
const props=defineProps({context:{type:String,required:true},specs:{type:Array,required:true},values:{type:Object,required:true},title:{type:String,default:'运行参数'},rowEditor:Boolean,extensions:{type:Object,default:()=>({})},overrides:{type:Object,default:()=>({})}})
const emit=defineEmits(['change','apply','update:overrides'])
const opened=ref(false), filter=ref(''), query=ref(''), draft=ref({}), rows=ref([])
const view=ref('common'), scrollArea=ref(null)
const adding=ref(false), newScope=ref(''), newName=ref(''), newText=ref(''), addError=ref('')
const clone=v=>v==null?v:JSON.parse(JSON.stringify(v))
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b)
const value=s=>props.values[s.key]===undefined?s.default:props.values[s.key]
const display=v=>v===undefined?'由底层决定':JSON.stringify(v)
const sampler=computed(()=>{
  const row=rows.value.find(r=>r.scope==='tuner'&&r.name==='sampler')
  try{return row?JSON.parse(row.text):'random'}catch{return 'random'}
})
const entries=computed(()=>{
  const entries=props.specs.map(spec=>({...spec,group:parameterGroup(spec),id:spec.key}))
  for(const [scope,group] of Object.entries(props.extensions)){
    const params=availableParameters(group,draft.value.method,sampler.value)
    for(const r of rows.value.filter(r=>r.scope===scope)){
      if(!Object.hasOwn(params,r.name))params[r.name]={}
    }
    for(const [key,spec] of Object.entries(params)){
      entries.push({...spec,id:scope+'.'+key,group:group.label,key,scope})
    }
  }
  return orderParameters(entries,props.context)
})
const groups=computed(()=>Array.from(new Set(entries.value.map(s=>s.group))))
const common=computed(()=>entries.value.filter(s=>s.common&&(!s.when||s.when(draft.value))))
const overridesById=computed(()=>new Map(rows.value.map(row=>[row.scope+'.'+row.name,row])))
const modified=computed(()=>entries.value.filter(s=>s.scope?overridesById.value.has(s.id):!same(draft.value[s.key],s.jsonValue?JSON.stringify(s.default):s.default)))
const views=computed(()=>[
  ...(common.value.length?[{id:'common',label:'常用参数',items:common.value}]:[]),
  {id:'all',label:'全部参数',items:entries.value},
  {id:'modified',label:'已调整',items:modified.value},
])
const visible=computed(()=>{
  const term=query.value.trim().toLowerCase()
  const items=views.value.find(v=>v.id===view.value)?.items??entries.value
  return items.filter(s=>(!filter.value||s.group===filter.value)&&(!term||[s.key,s.label,s.group,s.description].join(' ').toLowerCase().includes(term)))
})
watch([view,filter,query],()=>{if(scrollArea.value)scrollArea.value.scrollTop=0})
function selectView(id){view.value=id;filter.value='';query.value=''}
function search(value){query.value=value;if(value.trim())view.value='all'}
function selectGroup(value){filter.value=value;view.value='all'}
function defaultLabel(s){
  if(s.hint)return s.hint
  if(s.default===null&&!s.scope&&!s.jsonValue)return '未选择'
  if(s.type==='json')return s.default
  if(typeof s.default==='boolean')return s.default?'开启':'关闭'
  if(!s.scope&&s.choices&&!Array.isArray(s.default))return s.choices.find(c=>c.value===s.default)?.label??display(s.default)
  return display(s.default)
}
function setOverride(scope,name,text){
  const index=rows.value.findIndex(row=>row.scope===scope&&row.name===name)
  if(text===''){if(index!==-1)rows.value.splice(index,1);return}
  if(index===-1)rows.value.push({scope,name,text})
  else rows.value[index].text=text
}
const suggestions=computed(()=>availableParameters(props.extensions[newScope.value],draft.value.method,sampler.value))
function startAdding(){
  newScope.value=Object.keys(props.extensions).find(key=>props.extensions[key].label===filter.value)||Object.keys(props.extensions)[0]
  newName.value='';newText.value='';addError.value='';adding.value=true
}
function add(){
  const row={scope:newScope.value,name:newName.value.trim(),text:newText.value}
  try{parseRows([...rows.value,row],props.extensions,draft.value.method)}
  catch(e){addError.value=e.message;return}
  rows.value.push(row);selectGroup(props.extensions[row.scope].label);query.value=row.name;adding.value=false
}
const changed=computed(()=>props.specs.filter(s=>(!s.when||s.when(props.values))&&!same(value(s),s.default)))
const overrideCount=computed(()=>parameterRows(props.overrides).length)
const validation=computed(()=>{
  if(rows.value.some(r=>r.scope==='selector')&&draft.value.selector==='none')return '请先选择特征筛选方法'
  if(rows.value.some(r=>r.scope==='selector'&&['regularization','bin_params'].includes(r.name))&&draft.value.selector==='null')return 'regularization 和 bin_params 用于 IV 筛选'
  if(rows.value.some(r=>r.scope==='score_transform')&&draft.value.scorecard===false)return '请先启用概率转评分'
  if(rows.value.some(r=>r.scope==='importance')&&draft.value.explain===false)return '请启用 SHAP 与变量交互，或删除模型解释图覆盖值' 
  try{parseRows(rows.value,props.extensions,draft.value.method)}catch(e){return e.message}
  if(draft.value.min_n_bins>draft.value.max_n_bins)return '最少分箱数不能超过最多分箱数'
  if(draft.value.search && 'search_spaces' in draft.value){
    try{const obj=JSON.parse(draft.value.search_spaces);if(!obj||Array.isArray(obj)||typeof obj!=='object')return '搜索空间需要填写 JSON 对象'}
    catch{return '搜索空间不是有效的 JSON 对象'}
  }
  for(const s of props.specs.filter(s=>s.jsonValue)){
    try{const parsed=JSON.parse(draft.value[s.key]);if(parsed!==null&&typeof parsed!=='number')return s.label+' 需要数值或 null'}catch{return s.label+' 需要有效的 JSON 值'}
  }
  return ''
})
function edit(){
  draft.value=Object.fromEntries(props.specs.map(s=>[s.key,s.jsonValue?JSON.stringify(value(s)):clone(value(s))]))
  rows.value=parameterRows(props.overrides)
  selectView(common.value.length?'common':'all');adding.value=false;opened.value=true
}
function resetVisible(){
  for(const s of visible.value){
    if(s.scope)setOverride(s.scope,s.key,'')
    else draft.value[s.key]=s.jsonValue?JSON.stringify(s.default):clone(s.default)
  }
}
function apply(){
  if(validation.value)return
  const overrides=clone(parseRows(rows.value,props.extensions,draft.value.method))
  if(!same(overrides,props.overrides))emit('update:overrides',overrides)
  const values=clone(draft.value)
  for(const s of props.specs.filter(s=>s.jsonValue))values[s.key]=JSON.parse(draft.value[s.key])
  emit('apply',values,overrides)
  if(!props.rowEditor)for(const s of props.specs)if(!same(value(s),values[s.key]))emit('change',s.key,values[s.key])
  opened.value=false
}
watch(()=>props.specs.map(s=>s.key).join('|'),()=>{opened.value=false})
</script>
<template>
  <el-button v-if="rowEditor" link type="primary" :aria-label="'配置'+title" @click="edit">配置参数</el-button>
  <div v-else class="config-summary"><div><strong>{{title}}</strong><small>可选</small><span :title="changed.map(s=>s.label).join('、')">{{changed.length?'已调整 '+changed.slice(0,2).map(s=>s.label).join('、')+(changed.length>2?' 等 '+changed.length+' 项':''):overrideCount?'':'使用默认参数'}}{{overrideCount?' 底层参数 '+overrideCount+' 项':''}}</span></div>
    <el-button plain :icon="Setting" @click="edit">配置{{title}}</el-button></div>
  <el-dialog v-model="opened" :title="'配置'+title" class="parameter-dialog" width="min(1060px, 96vw)" align-center append-to-body destroy-on-close :close-on-click-modal="false">
    <div class="parameter-views" role="group" aria-label="参数范围">
      <button v-for="v in views" :key="v.id" type="button" :aria-pressed="v.id===view" @click="selectView(v.id)">{{v.label}} <span>{{v.items.length}}</span></button>
    </div>
    <div class="params-toolbar">
      <el-input :model-value="query" @update:model-value="search" :prefix-icon="Search" clearable placeholder="搜索全部参数名或说明" aria-label="搜索参数"/>
      <el-select :model-value="filter" @update:model-value="selectGroup" placeholder="全部分组" aria-label="筛选参数分组">
        <el-option label="全部分组" value=""/><el-option v-for="name in groups" :key="name" :value="name" :label="name"/>
      </el-select>
      <span v-if="query||filter" role="status">{{visible.length}} 项</span>
      <el-button v-if="Object.keys(extensions).length" @click="startAdding">添加自定义参数</el-button>
    </div>
    <div v-if="adding" class="custom-parameter">
      <el-select v-model="newScope" aria-label="自定义参数用途" @change="newName='';newText='';addError=''"><el-option v-for="(g,key) in extensions" :key="key" :label="g.label" :value="key"/></el-select>
      <el-select v-model="newName" aria-label="自定义参数名" placeholder="选择或输入参数名" filterable allow-create default-first-option @change="name=>{newText=JSON.stringify(suggestions[name]?.default)??'';addError=''}">
        <el-option v-for="(_,key) in suggestions" :key="key" :value="key" :label="key"/>
      </el-select>
      <el-input v-model="newText" aria-label="自定义参数值" placeholder='JSON 值，如 false 或 "auto"'/>
      <el-button @click="add">添加</el-button><el-button link @click="adding=false">取消添加</el-button>
      <small v-if="addError" role="alert">{{addError}}</small>
    </div>
    <div ref="scrollArea" class="params-scroll">
      <table class="params-table" :aria-label="title+'总表'">
        <thead><tr><th scope="col">参数</th><th scope="col">作用说明</th><th scope="col">配置值</th></tr></thead>
        <tbody><tr v-for="s in visible" :key="s.id">
          <th scope="row"><strong>{{s.label||s.key}}</strong><code v-if="s.label&&s.label!==(s.apiName||s.key)">{{s.apiName||s.key}}</code>
            <small>{{s.group}}</small></th>
          <td class="param-description">
            <el-tooltip v-if="s.description?.length>90" placement="top" :show-after="180" popper-class="parameter-help-popper" :content="s.description">
              <span class="param-help-preview" tabindex="0">{{s.description}}</span>
            </el-tooltip>
            <template v-else>{{s.description||'当前接口未提供说明'}}</template>
          </td>
          <td class="param-control"><div v-if="s.scope" class="params-value">
            <el-select v-if="s.choices" :model-value="overridesById.has(s.id)?JSON.parse(overridesById.get(s.id).text):s.default" :aria-label="s.group+' '+s.key" @change="v=>setOverride(s.scope,s.key,JSON.stringify(v))">
              <el-option v-for="v in s.choices" :key="String(v)" :label="v===null?'默认':String(v)" :value="v"/>
            </el-select>
            <el-input v-else :model-value="overridesById.get(s.id)?.text??''" :placeholder="display(s.default)" :aria-label="s.group+' '+s.key" @update:model-value="v=>setOverride(s.scope,s.key,v)"/>
            <el-button v-if="overridesById.has(s.id)" link :aria-label="'删除 '+s.key+' 覆盖值'" @click="setOverride(s.scope,s.key,'')">删除</el-button>
          </div>
          <span v-else-if="s.when&&!s.when(draft)" class="muted">{{display(draft[s.key])}} · 未启用</span>
          <FieldSelect v-else-if="s.fields" v-model="draft[s.key]" :fields="s.fields" :multiple="s.multi" :limit="s.limit" :label="s.label" clearable/>
          <el-select v-else-if="s.choices" :class="{'scrollable-tags':s.multi}" v-model="draft[s.key]" :aria-label="s.label" :multiple="s.multi" filterable :allow-create="s.custom" default-first-option><el-option v-for="c in s.choices" :key="c.value" :value="c.value" :label="c.label"/></el-select>
          <CodeEditor v-else-if="s.type==='json'" v-model="draft[s.key]" :label="s.label" :schema="s.schema" :rows="4"/>
          <el-input v-else-if="s.jsonValue" v-model="draft[s.key]" :aria-label="s.label"/>
          <el-switch v-else-if="typeof s.default==='boolean'" v-model="draft[s.key]" :aria-label="s.label"/>
          <el-input-number v-else-if="typeof s.default==='number'" :model-value="draft[s.key]" :aria-label="s.label" :controls="false" :min="s.min" :max="s.max" :step="s.step||1" @update:model-value="v=>draft[s.key]=v??s.default"/>
          <el-input v-else v-model="draft[s.key]" :aria-label="s.label"/>
          <small class="param-default">默认值 · {{defaultLabel(s)}}</small></td>
        </tr></tbody>
      </table>
      <p v-if="!visible.length" class="params-empty">{{view==='modified'?'暂无已调整的参数':'没有匹配的参数'}}</p>
    </div>
    <small v-if="visible.some(s=>s.scope&&!s.choices)||adding" class="params-note muted">底层参数使用 JSON，文本加双引号。清空覆盖值后使用默认值。</small>
    <el-alert v-if="validation" :title="validation" type="error" :closable="false"/>
    <template #footer><div class="parameter-footer"><el-button link :disabled="!visible.length" @click="resetVisible">恢复当前列表默认值</el-button><span/><el-button @click="opened=false">取消</el-button><el-button type="primary" :disabled="!!validation||adding" @click="apply">应用配置</el-button></div></template>
  </el-dialog>
</template>

<style scoped>
.parameter-views{display:flex;gap:18px;border-bottom:1px solid var(--border);margin-bottom:12px}
.parameter-views button{display:inline-flex;gap:6px;align-items:baseline;border:0;border-bottom:2px solid transparent;background:transparent;padding:4px 2px 9px;color:var(--muted);font:inherit;font-weight:600;cursor:pointer}
.parameter-views button[aria-pressed=true]{color:var(--accent-text);border-bottom-color:currentColor}
.parameter-views button span{font-size:12px;font-variant-numeric:tabular-nums}
.params-toolbar,.custom-parameter,.params-value{display:flex;align-items:center;gap:8px}
.params-toolbar{margin-bottom:12px;flex-wrap:wrap}
.params-toolbar>.el-input{width:min(240px,100%)}.params-toolbar>.el-select{width:155px}
.params-toolbar>span{color:var(--muted);font-size:12px}.params-toolbar>.el-button{margin-left:auto}
.custom-parameter{flex-wrap:wrap;margin-bottom:12px}.custom-parameter>.el-select,.custom-parameter>.el-input{flex:1;min-width:125px}
.custom-parameter small{width:100%;color:var(--el-color-danger)}
.params-scroll{overflow:auto;min-height:0;border:1px solid var(--border);border-radius:var(--radius)}
.params-table{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;text-align:left;font-size:13px}
.params-table th,.params-table td{padding:8px 12px;overflow-wrap:anywhere;border-bottom:1px solid var(--line-soft);vertical-align:middle}
.params-table tbody tr:last-child>:is(th,td){border-bottom:0}
.params-table thead th{position:sticky;top:0;z-index:2;background:var(--table-head);color:var(--muted);font-size:12px}
.params-table thead th:first-child{width:25%}.params-table thead th:nth-child(2){width:39%}
.params-table tbody th{font-weight:400}.params-table th>strong,.params-table th>code,.params-table th>small,.param-default{display:block}
.params-table th>code,.params-table small{color:var(--muted);font-size:11px;line-height:1.5}
.param-description{line-height:1.6}.param-default{margin-top:3px}.params-table .el-input-number{width:100%}
.param-help-preview{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;cursor:help}
.params-note{margin-top:6px}.params-empty{padding:14px;text-align:center;color:var(--muted)}
.parameter-footer{width:100%}
@media(max-width:680px){
  .parameter-views{gap:12px}.params-toolbar>.el-input,.params-toolbar>.el-select{width:auto;flex:1 1 125px}
  .params-table,.params-table tbody{display:block}.params-table thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
  .params-table tbody tr{display:grid;gap:5px;padding:10px 12px;border-bottom:1px solid var(--border)}
  .params-table tbody th,.params-table tbody td{display:block;border:0;padding:0}.params-table tbody th>:is(strong,code,small){display:inline;margin-right:7px}
  .param-help-preview{display:block}
}
</style>

<style>
.parameter-dialog.el-dialog{display:flex;flex-direction:column;max-height:92vh;max-height:92dvh}
.parameter-dialog>.el-dialog__header{margin-bottom:12px;padding-bottom:10px}
.parameter-dialog>.el-dialog__body{display:flex;flex-direction:column;min-height:0;overflow:auto}
.parameter-dialog>.el-dialog__footer{padding-top:10px;margin-top:10px}
.parameter-help-popper{max-width:min(520px,90vw);white-space:pre-line;line-height:1.55}
</style>
