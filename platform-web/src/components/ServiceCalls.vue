<script setup>
import {computed, onMounted, ref} from 'vue'
import {api} from '../api'

const props = defineProps({service: {type: Object, required: true}})
const emit = defineEmits(['close'])
const endpoint = '/services/' + props.service.id
const rows = ref([]), total = ref(0), page = ref(1)
const failedOnly = ref(false), search = ref(''), appliedSearch = ref('')
const loading = ref(false), error = ref(''), ready = ref(false)
const retention = ref(null), settingsOpen = ref(false), mode = ref('forever'), days = ref(30)
const saving = ref(false), settingsError = ref('')
const selected = ref(''), detail = ref(null), detailError = ref(''), bodyTab = ref('request')
const shorten = computed(() => mode.value === 'days' && (retention.value === null || days.value < retention.value))
const validDays = computed(() => Number.isInteger(days.value) && days.value >= 1 && days.value <= 36500)
const stamp = seconds => new Date(seconds * 1000).toLocaleString('zh-CN', {hour12:false})
let generation = 0

async function load(number = 1) {
  const current = ++generation
  page.value = number
  loading.value = true
  error.value = ''
  try {
    const params = new URLSearchParams({page:number, failed_only:failedOnly.value, call_id:appliedSearch.value})
    const result = await api(endpoint + '/calls?' + params)
    if (current !== generation) return
    rows.value = result.items
    total.value = result.total
    retention.value = result.retention_days
    ready.value = true
  } catch (e) {
    if (current === generation) error.value = e.message
  } finally {
    if (current === generation) loading.value = false
  }
}

function find() {
  appliedSearch.value = search.value.trim()
  load()
}

async function inspect(id) {
  selected.value = id
  detail.value = null
  detailError.value = ''
  bodyTab.value = 'request'
  try {
    const result = await api(endpoint + '/calls/' + id)
    if (selected.value === id) detail.value = result
  } catch (e) {
    if (selected.value === id) detailError.value = e.message
  }
}

function editRetention() {
  mode.value = retention.value === null ? 'forever' : 'days'
  days.value = retention.value ?? 30
  settingsError.value = ''
  settingsOpen.value = true
}

async function save() {
  saving.value = true
  try {
    const value = mode.value === 'forever' ? null : days.value
    await api(endpoint + '/retention', {method:'PUT', body:JSON.stringify({retention_days:value})})
    retention.value = value
    settingsOpen.value = false
    await load()
  } catch (e) {
    settingsError.value = e.message
  } finally {
    saving.value = false
  }
}

onMounted(() => load())
</script>

<template>
  <el-drawer :model-value="true" @close="emit('close')" :title="service.name+' · 调用记录'" size="min(950px, 98vw)">
    <div class="call-toolbar">
      <el-input v-model="search" clearable maxlength="32" placeholder="输入完整调用编号" aria-label="调用编号" @keyup.enter="find" @clear="find"/>
      <el-button @click="find">查询</el-button>
      <el-checkbox v-model="failedOnly" @change="load()">仅看失败</el-checkbox>
      <el-button @click="load(page)">刷新</el-button>
      <el-button v-if="ready" link type="primary" @click="editRetention">保存期限 · {{retention === null ? '永久' : retention+' 天'}}</el-button>
    </div>
    <el-alert v-if="error" type="error" :title="error" :closable="false"/>
    <el-table v-loading="loading" :data="rows" stripe empty-text="暂无匹配的调用记录">
      <el-table-column label="调用时间" min-width="166"><template #default="{row}">{{stamp(row.started_at)}}</template></el-table-column>
      <el-table-column label="HTTP 状态" width="110"><template #default="{row}"><el-tag :type="row.status >= 400 ? 'danger' : 'success'">{{row.status}}</el-tag></template></el-table-column>
      <el-table-column label="耗时" width="105"><template #default="{row}">{{row.duration_ms.toFixed(1)}} ms</template></el-table-column>
      <el-table-column prop="id" label="调用编号" min-width="190" show-overflow-tooltip/>
      <el-table-column width="82"><template #default="{row}"><el-button link type="primary" @click="inspect(row.id)">详情</el-button></template></el-table-column>
    </el-table>
    <el-pagination v-if="total" :current-page="page" :page-size="20" :total="total" layout="total, prev, pager, next" @current-change="load"/>
  </el-drawer>
  <el-dialog v-model="settingsOpen" title="调用记录保存期限" width="min(460px, 94vw)" append-to-body>
    <el-radio-group v-model="mode" aria-label="保存方式"><el-radio-button value="forever">永久保存</el-radio-button><el-radio-button value="days">按天保存</el-radio-button></el-radio-group>
    <el-form v-if="mode==='days'" label-position="top" class="retention-days"><el-form-item label="保留最近多少天"><el-input-number v-model="days" :min="1" :max="36500" :precision="0" :controls="false"/></el-form-item></el-form>
    <p v-if="shorten" class="muted">保存后将立即删除本服务超期的历史记录，删除后无法恢复。</p>
    <el-alert v-if="settingsError" type="error" :title="settingsError" :closable="false"/>
    <template #footer><el-button @click="settingsOpen=false">取消</el-button><el-button type="primary" :loading="saving" :disabled="mode==='days' && !validDays" @click="save">保存期限</el-button></template>
  </el-dialog>
  <el-dialog :model-value="!!selected" @close="selected=''" title="调用详情" width="min(900px, 96vw)" top="5vh" append-to-body>
    <el-alert v-if="detailError" type="error" :title="detailError" :closable="false"/>
    <template v-else-if="detail">
      <p><code>{{detail.method}} {{detail.target}}</code></p>
      <p class="muted">{{stamp(detail.started_at)}} · HTTP {{detail.status}} · {{detail.duration_ms.toFixed(1)}} ms · {{detail.id}}</p>
      <el-tabs v-model="bodyTab"><el-tab-pane label="请求" name="request"/><el-tab-pane label="响应" name="response"/></el-tabs>
      <p v-if="detail[bodyTab+'_body'].encoding==='base64'" class="muted">二进制原文以 Base64 展示</p>
      <pre class="call-body">{{detail[bodyTab+'_body'].text || '（空正文）'}}</pre>
      <el-collapse><el-collapse-item title="HTTP 头"><pre class="call-body">{{detail[bodyTab+'_headers'].map(([k,v])=>k+': '+v).join('\n')}}</pre></el-collapse-item></el-collapse>
    </template>
    <el-skeleton v-else :rows="4" animated/>
  </el-dialog>
</template>

<style scoped>
.call-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.call-toolbar .el-input{width:min(270px,100%)}.call-toolbar .el-button+.el-button{margin-left:0}
.retention-days,.el-pagination{margin-top:14px}
.call-body{max-height:48vh;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;border:1px solid var(--border);background:var(--surface);color:var(--text)}
</style>
