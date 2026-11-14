<script setup>
import TableFrame from './TableFrame.vue'
import {requestSchema, rulesSchema} from '../editor-context'
import CodeEditor from './CodeEditor.vue'
import ServiceCalls from './ServiceCalls.vue'
import {ref,computed,onMounted,watch} from 'vue'
import {readDraft,rememberForm,writeDraft} from '../form-drafts'
import {api} from '../api'
const props = defineProps({dataset:Object, datasets:Array})
const fieldSource = ref()
watch(() => props.dataset?.id, (id,previous) => {
  if(fieldSource.value===previous || !props.datasets?.some(row=>row.id===fieldSource.value))fieldSource.value=id
},{immediate:true})
const ruleFields = computed(() => props.datasets?.find(d => d.id === fieldSource.value)?.columns || [])
const publishing=ref(false), callService=ref(null), lastCallId=ref('')
const services=ref([]),jobs=ref([]),source=ref('job'),name=ref(''),jobId=ref(''),file=ref('model.joblib'),upload=ref(null),trusted=ref(false),logic=ref('or'),rulesText=ref('[{"name":"高风险","expression":"评分 < 500"}]'),error=ref(''),busy=ref(false),issued=ref(null),schema=ref(null),active=ref(null),callToken=ref(''),requestText=ref('{"records":[]}'),response=ref('')
const bodySchema = computed(() => requestSchema(active.value?.features || []))
rememberForm('publish-service',
  () => ({source:source.value,name:name.value,jobId:jobId.value,file:file.value,logic:logic.value,rulesText:rulesText.value,fieldSource:fieldSource.value}),
  saved => {
    source.value=saved.source;name.value=saved.name;jobId.value=saved.jobId
    file.value=saved.file;logic.value=saved.logic;rulesText.value=saved.rulesText
    if(saved.fieldSource!==undefined)fieldSource.value=saved.fieldSource
  })
const selected=computed(()=>jobs.value.find(j=>j.id===jobId.value))
async function refresh(){try{services.value=await api('/services');jobs.value=(await api('/jobs')).filter(j=>j.status==='成功'&&j.artifacts.some(f=>f.startsWith('model')&&f.endsWith('.joblib')))}catch(e){error.value=e.message}}
async function publish(){busy.value=true;error.value='';try{if(source.value==='upload'){const body=new FormData();body.append('file',upload.value);body.append('name',name.value);body.append('trusted',String(trusted.value));issued.value=await api('/services/upload',{method:'POST',body})}else{issued.value=await api('/services',{method:'POST',body:JSON.stringify({name:name.value,...(source.value==='job'?{source_job:jobId.value,model_file:file.value}:{rules:JSON.parse(rulesText.value),logic:logic.value})})})}callToken.value=issued.value.token;publishing.value=false;await refresh();await docs(issued.value)}catch(e){error.value=e.message}finally{busy.value=false}}
async function toggle(row){try{await api(`/services/${row.id}`,{method:'PATCH',body:JSON.stringify({enabled:!row.enabled})});await refresh()}catch(e){error.value=e.message}}
async function rotate(row){try{const result=await api(`/services/${row.id}/rotate-token`,{method:'POST'});issued.value={...row,...result};await docs(issued.value)}catch(e){error.value=e.message}}
let docRequest=0,testRequest=0
watch(requestText,text=>{if(active.value)writeDraft('service-request/'+active.value.id,text)}, {flush:'sync'})
async function docs(row){
  const current=++docRequest
  if(active.value?.id!==row.id){
    testRequest++
    active.value=row
    schema.value=null
    callToken.value=issued.value?.id===row.id?issued.value.token:''
    requestText.value=readDraft('service-request/'+row.id) ?? JSON.stringify({records:[Object.fromEntries(row.features.map(f=>[f,0]))]},null,2)
    response.value='';lastCallId.value=''
  }else if(issued.value?.id===row.id)callToken.value=issued.value.token
  try{
    const loaded=await api(`/services/${row.id}/schema`)
    if(current===docRequest)schema.value=loaded
  }catch(e){if(current===docRequest)error.value=e.message}
}
async function test(){
  const current=++testRequest
  try{
    const body=JSON.stringify(JSON.parse(requestText.value))
    const res=await fetch(active.value.endpoint,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${callToken.value}`},body})
    const content=await res.text()
    if(current===testRequest){lastCallId.value=res.headers.get('X-Request-ID')||'';response.value=content}
  }catch(e){if(current===testRequest)error.value=e.message}
}
function exportSchema(){const url=URL.createObjectURL(new Blob([JSON.stringify(schema.value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='openapi.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
onMounted(refresh)
</script>
<template>
  <el-alert v-if="error" :title="error" type="error" closable @close="error=''"/>
  <section class="panel"><div class="panel-heading"><h2>已发布服务</h2><div><el-button @click="refresh">刷新</el-button><el-button type="primary" @click="publishing=true">发布服务</el-button></div></div><TableFrame v-slot="{tableSize}">
  <el-table v-bind="tableSize" :data="services"><el-table-column prop="name" label="服务名称"/><el-table-column label="状态"><template #default="{row}"><el-tag :type="row.enabled?'success':'info'">{{row.enabled?'已上线':'已下线'}}</el-tag></template></el-table-column><el-table-column prop="endpoint" label="预测API" min-width="224"/><el-table-column label="管理" min-width="320"><template #default="{row}"><el-button link type="primary" @click="callService=row">调用记录</el-button><el-button link @click="docs(row)">文档与调试</el-button><el-button link @click="toggle(row)">{{row.enabled?'下线':'上线'}}</el-button><el-button link @click="rotate(row)">重新生成令牌</el-button></template></el-table-column></el-table>
</TableFrame></section>
  <el-dialog v-model="publishing" title="发布推理服务" width="min(760px, 95vw)" :close-on-click-modal="!busy"><el-alert v-if="error" :title="error" type="error" :closable="false"/>
    <el-form @submit.prevent label-position="top"><div class="form-grid"><el-form-item label="服务名称"><el-input v-model="name" maxlength="100"/></el-form-item><el-form-item label="发布来源"><el-select v-model="source"><el-option label="策略／模型任务产物" value="job"/><el-option label="手工上传 hscredit 模型" value="upload"/><el-option label="手写 RuleSet" value="rules"/></el-select></el-form-item>
      <template v-if="source==='job'"><el-form-item label="成功任务"><el-select v-model="jobId" filterable><el-option v-for="j in jobs" :key="j.id" :value="j.id" :label="`${j.dataset_name} · ${j.created_at}`"/></el-select></el-form-item><el-form-item label="模型文件"><el-select v-model="file"><el-option v-for="f in selected?.artifacts.filter(f=>f.startsWith('model')&&f.endsWith('.joblib'))||[]" :key="f" :value="f"/></el-select></el-form-item></template>
      <template v-if="source==='upload'"><el-form-item label="模型文件（joblib / pickle，100 MB以内）"><input type="file" accept=".joblib,.pkl,.pickle" @change="upload=$event.target.files[0]"/></el-form-item><el-form-item label="文件来源"><el-checkbox v-model="trusted">我确认文件来自可信来源</el-checkbox><span class="form-help">pickle/joblib 反序列化可以执行代码，请仅导入您信任的模型。</span></el-form-item></template>
      <template v-if="source==='rules'"><el-form-item label="字段提示来源"><el-select v-model="fieldSource" clearable filterable placeholder="选择参考数据集"><el-option v-for="d in datasets" :key="d.id" :value="d.id" :label="d.name"/></el-select></el-form-item><el-form-item label="组合逻辑"><el-select v-model="logic"><el-option label="任一命中" value="or"/><el-option label="全部命中" value="and"/></el-select></el-form-item><el-form-item class="full-width" label="RuleSet 规则（JSON数组，包含 name 和 expression）"><CodeEditor v-model="rulesText" :schema="rulesSchema" :expression-context="{fields:ruleFields, embedded:true}" label="RuleSet 规则" :rows="5"/></el-form-item></template>
    </div><div class="submit-bar"><p class="muted">发布到当前 FastAPI 实例，使用相同地址和端口；每个服务独立生成令牌。</p><el-button type="primary" :loading="busy" :disabled="!name||(source==='job'&&!jobId)||(source==='upload'&&(!upload||!trusted))" @click="publish">上线服务</el-button></div></el-form>
  </el-dialog>
  <section v-if="issued" class="panel"><h2>服务令牌</h2><p class="muted">令牌仅在生成时显示，请保存；服务列表不返回明文。重新生成后旧令牌立即失效。</p><el-input :model-value="issued.token" readonly aria-label="新服务令牌"/><code>Authorization: Bearer {{issued.token}}</code></section>

  <section v-if="schema" class="panel"><div class="panel-heading"><h2>{{active.name}} · API文档</h2><el-button @click="exportSchema">下载 OpenAPI 文档</el-button></div><p><code>POST {{active.endpoint}}</code> · JSON 请求 · 每次最多 10000 行</p><p>必填字段：{{active.features.join('、')||'由模型校验'}}。返回顺序与输入顺序一致；概率数组的类别顺序见 classes。</p><p><code>GET /services/{{active.id}}/openapi.json</code> 同样使用服务令牌鉴权。</p><div class="form-grid"><label>Authorization Bearer 令牌<el-input v-model="callToken" type="password" show-password/></label><div class="full-width">请求体<CodeEditor v-model="requestText" :schema="bodySchema" label="请求体" :rows="6"/></div></div><div class="download-bar"><el-button @click="test">发送测试请求</el-button></div><p v-if="lastCallId" class="muted">调用编号 {{lastCallId}}</p><pre v-if="response">{{response}}</pre><el-collapse><el-collapse-item title="OpenAPI Schema"><pre>{{JSON.stringify(schema,null,2)}}</pre></el-collapse-item></el-collapse></section>
  <ServiceCalls v-if="callService" :key="callService.id" :service="callService" @close="callService=null"/>
</template>
