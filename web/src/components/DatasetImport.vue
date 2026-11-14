<script setup>
import {computed, reactive, ref, shallowRef, watch} from 'vue'
import {Upload} from '@element-plus/icons-vue'
import {api} from '../api'
import {rememberForm} from '../form-drafts'

const open = defineModel({type:Boolean})
const emit = defineEmits(['imported'])
const source = ref('file'), name = ref(''), file = ref(null), fileInput = ref(null)
const encoding = ref('utf-8-sig'), sheet = ref(''), importing = ref(false), testing = ref(false), error = ref('')
const busy = computed(() => importing.value || testing.value)
const engines = {mysql:'MySQL / MariaDB', starrocks:'StarRocks', clickhouse:'ClickHouse', hive:'Hive', impala:'Impala', oracle:'Oracle', maxcompute:'MaxCompute'}
const ports = {mysql:3306, starrocks:9030, clickhouse:8123, hive:10000, impala:21050}
const engine = ref('mysql'), sql = ref('')
const connection = reactive({address:'', port:3306, database:'', username:'', password:'', secure:false, auth_mechanism:'PLAIN'})
const hive = computed(() => ['hive','impala'].includes(engine.value))
const server = computed(() => Object.hasOwn(ports, engine.value))
const addressLabel = computed(() => ({oracle:'连接串（DSN）', maxcompute:'服务地址（Endpoint）'}[engine.value] || '主机地址'))
const addressHint = computed(() => ({oracle:'host:1521/service_name', maxcompute:'https://…/api'}[engine.value] || '主机名或 IP'))
const useTunnel = ref(false), keyFile = shallowRef(null), keyInput = ref(null), testMessage = ref('')
const emptyTunnel = () => ({host:'', port:22, username:'', auth:'password', password:'', passphrase:'', host_fingerprint:''})
const tunnel = reactive(emptyTunnel())
const needsKey = computed(() => tunnel.auth !== 'password')
const needsPassword = computed(() => tunnel.auth !== 'key')
const canConnect = computed(() => connection.address.trim() && (engine.value !== 'maxcompute' || connection.database.trim()) &&
  (!useTunnel.value || (tunnel.host.trim() && tunnel.port && tunnel.username.trim() && (!needsKey.value || keyFile.value) && (!needsPassword.value || tunnel.password))))
const canSubmit = computed(() => source.value === 'file' ? !!file.value : !!(canConnect.value && name.value.trim() && sql.value.trim()))

function resetTunnel(){
  Object.assign(tunnel, emptyTunnel())
  keyFile.value = null
}
watch(useTunnel, enabled => { if (!enabled) resetTunnel() })
watch(() => tunnel.auth, () => {
  if (!needsPassword.value) tunnel.password = ''
  if (!needsKey.value) { keyFile.value = null; tunnel.passphrase = '' }
})
watch([engine, connection, useTunnel, tunnel, keyFile, source], () => {
  testMessage.value = error.value = ''
})
async function connectionBody(){
  const body = {db_type:engine.value, ...connection}
  if (!hive.value) delete body.auth_mechanism
  if (useTunnel.value) {
    body.tunnel = {host:tunnel.host.trim(), port:tunnel.port, username:tunnel.username.trim(), auth:tunnel.auth, host_fingerprint:tunnel.host_fingerprint.trim()}
    if (needsPassword.value) body.tunnel.password = tunnel.password
    if (needsKey.value) {
      body.tunnel.private_key = await keyFile.value.text()
      body.tunnel.passphrase = tunnel.passphrase
    }
  }
  return body
}
async function testConnection(){
  if (busy.value || !canConnect.value) return
  testing.value = true
  testMessage.value = error.value = ''
  try {
    const result = await api('/datasets/database/test', {method:'POST', body:JSON.stringify(await connectionBody())})
    testMessage.value = result.message
  } catch (e) {
    error.value = e.message
  } finally {
    testing.value = false
  }
}

function resetConnection() {
  Object.assign(connection, {address:'', port:ports[engine.value], database:'', username:'', password:'', secure:false, auth_mechanism:engine.value === 'impala' ? 'NOSASL' : 'PLAIN'})
}
watch(engine, resetConnection,{flush:'sync'})
watch(() => connection.secure, secure => {
  if (engine.value === 'clickhouse' && [8123,8443].includes(connection.port)) connection.port = secure ? 8443 : 8123
})
rememberForm('dataset-import', () => ({
  source:source.value,name:name.value,sheet:sheet.value,encoding:encoding.value,engine:engine.value,sql:sql.value,
  connection:{address:connection.address,port:connection.port,database:connection.database,secure:connection.secure,auth_mechanism:connection.auth_mechanism},
  useTunnel:useTunnel.value,tunnel:{host:tunnel.host,port:tunnel.port,auth:tunnel.auth,host_fingerprint:tunnel.host_fingerprint},
}), saved => {
  source.value=saved.source;name.value=saved.name;sheet.value=saved.sheet;encoding.value=saved.encoding
  engine.value=saved.engine;sql.value=saved.sql;useTunnel.value=saved.useTunnel
  Object.assign(connection,saved.connection)
  Object.assign(tunnel,saved.tunnel)
})
watch(open, visible => {
  if (!visible) {
    connection.password = tunnel.password = tunnel.passphrase = ''
    keyFile.value = null
    testMessage.value = error.value = ''
  }
})
async function submit() {
  if (busy.value || !canSubmit.value) return
  importing.value = true
  error.value = testMessage.value = ''
  try {
    let dataset
    if (source.value === 'file') {
      const body = new FormData()
      body.append('file',file.value)
      body.append('name',name.value)
      body.append('encoding',encoding.value)
      body.append('sheet',sheet.value)
      dataset = await api('/datasets',{method:'POST',body})
    } else {
      const body = {...await connectionBody(), name:name.value.trim(), sql:sql.value}
      dataset = await api('/datasets/database',{method:'POST',body:JSON.stringify(body)})
    }
    file.value = null
    open.value = false
    emit('imported',dataset)
  } catch (e) {
    error.value = e.message
  } finally {
    importing.value = false
  }
}
</script>

<template>
  <el-dialog v-model="open" title="导入数据集" width="min(640px, 94vw)" destroy-on-close
    :close-on-click-modal="!busy" :close-on-press-escape="!busy" :show-close="!busy">
    <el-tabs v-model="source">
      <el-tab-pane label="本地文件" name="file" :disabled="busy"/>
      <el-tab-pane label="数据库查询" name="database" :disabled="busy"/>
    </el-tabs>
    <el-form label-position="top" :disabled="busy" @submit.prevent="submit">
      <template v-if="source === 'file'">
        <div class="upload-zone" role="button" tabindex="0" @keydown.enter.prevent="!busy && fileInput.click()"
          @keydown.space.prevent="!busy && fileInput.click()" @click="!busy && fileInput.click()"
          @dragover.prevent @drop.prevent="!busy && (file=$event.dataTransfer.files[0])">
          <el-icon><Upload/></el-icon><strong>{{file?.name || '点击选择或拖入文件'}}</strong>
          <span>CSV、XLSX、Parquet · 单文件上限 100 MB</span>
          <input ref="fileInput" type="file" accept=".csv,.xlsx,.parquet" hidden @click.stop @change="file=$event.target.files[0]"/>
        </div>
        <div class="form-grid import-fields">
          <el-form-item label="CSV 文件编码"><el-select v-model="encoding"><el-option label="UTF-8" value="utf-8-sig"/><el-option label="GB18030 / GBK" value="gb18030"/></el-select></el-form-item>
          <el-form-item label="Excel 工作表"><el-input v-model="sheet" placeholder="留空读取第一张表"/></el-form-item>
        </div>
      </template>
      <template v-else>
        <el-form-item label="数据库类型"><el-select v-model="engine"><el-option v-for="(label,key) in engines" :key="key" :label="label" :value="key"/></el-select></el-form-item>
        <div class="form-grid import-fields">
          <el-form-item :label="addressLabel"><el-input v-model="connection.address" :placeholder="addressHint"/></el-form-item>
          <el-form-item v-if="server" label="端口"><el-input-number v-model="connection.port" :min="1" :max="65535" :controls="false"/></el-form-item>
          <el-form-item v-if="engine !== 'oracle'" :label="engine === 'maxcompute' ? '项目名' : '数据库名'"><el-input v-model="connection.database"/></el-form-item>
          <el-form-item :label="engine === 'maxcompute' ? 'AccessKey ID' : '用户名'"><el-input v-model="connection.username" placeholder="查询账号"/></el-form-item>
          <el-form-item :label="engine === 'maxcompute' ? 'AccessKey Secret' : '密码'"><el-input v-model="connection.password" type="password" show-password autocomplete="off"/></el-form-item>
          <el-form-item v-if="hive" label="认证方式"><el-select v-model="connection.auth_mechanism"><el-option v-for="mode in ['PLAIN','LDAP','NOSASL','GSSAPI']" :key="mode" :value="mode"/></el-select></el-form-item>
          <el-form-item v-if="server" label="TLS 加密连接"><el-switch v-model="connection.secure"/></el-form-item>
        </div>
        <el-form-item label="使用 SSH 隧道"><el-switch v-model="useTunnel"/></el-form-item>
        <fieldset v-if="useTunnel" class="ssh-settings">
          <legend>SSH 跳板机</legend>
          <div class="form-grid import-fields">
            <el-form-item label="SSH 主机地址"><el-input v-model="tunnel.host" placeholder="跳板机主机名或 IP"/></el-form-item>
            <el-form-item label="SSH 端口"><el-input-number v-model="tunnel.port" :min="1" :max="65535" :controls="false"/></el-form-item>
            <el-form-item label="SSH 用户名"><el-input v-model="tunnel.username"/></el-form-item>
            <el-form-item label="SSH 认证方式"><el-select v-model="tunnel.auth">
              <el-option label="账号密码" value="password"/><el-option label="公钥" value="key"/><el-option label="账号密码 + 公钥" value="key_password"/>
            </el-select></el-form-item>
            <el-form-item v-if="needsPassword" label="SSH 登录密码"><el-input v-model="tunnel.password" type="password" show-password autocomplete="off"/></el-form-item>
            <el-form-item v-if="needsKey" label="SSH 私钥文件">
              <el-button @click="keyInput.click()">{{keyFile?.name || '选择与公钥配对的私钥'}}</el-button>
              <input ref="keyInput" type="file" hidden @change="keyFile=$event.target.files[0]"/>
            </el-form-item>
            <el-form-item v-if="needsKey" label="私钥口令"><el-input v-model="tunnel.passphrase" type="password" show-password autocomplete="off" placeholder="仅加密私钥需要"/></el-form-item>
          </div>
          <el-form-item label="SSH 主机指纹"><el-input v-model="tunnel.host_fingerprint" placeholder="SHA256:…，留空使用平台用户的 known_hosts"/></el-form-item>
        </fieldset>
        <div class="connection-test">
          <el-button :disabled="!canConnect || importing" :loading="testing" @click="testConnection">测试连接</el-button>
          <span v-if="testMessage" role="status">{{testMessage}}</span>
        </div>
        <el-form-item label="查询 SQL"><el-input v-model="sql" type="textarea" :rows="4" placeholder="SELECT customer_id, score FROM risk_scores"/></el-form-item>
        <p class="muted import-hint">使用只读账号。查询结果保存为本次分析的数据集快照，连接信息仅用于此次导入。</p>
      </template>
      <el-form-item label="数据集名称"><el-input v-model="name" :placeholder="source === 'file' ? '留空使用文件名' : '例如：授信评分样本'" maxlength="150"/></el-form-item>
    </el-form>
    <el-alert v-if="error" type="error" :closable="false" :title="error" show-icon/>
    <template #footer>
      <el-button :disabled="busy" @click="open=false">取消</el-button>
      <el-button type="primary" :disabled="!canSubmit || testing" :loading="importing" @click="submit">{{importing ? '正在导入' : '导入并预览'}}</el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.import-fields{grid-template-columns:repeat(2,minmax(0,1fr))}
.import-fields .el-input-number{width:100%}
.import-hint{margin:0 0 16px}
.ssh-settings{min-width:0;margin:0 0 16px;padding:12px 14px 0;border:1px solid var(--el-border-color);border-radius:var(--radius)}
.ssh-settings legend{padding:0 6px;font-weight:600}
.connection-test{display:flex;align-items:center;gap:12px;margin:0 0 16px}
.connection-test span{color:var(--el-color-success)}
@media(max-width:520px){.import-fields{grid-template-columns:1fr}}
</style>
