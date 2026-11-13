<script setup>
import {computed, ref, watch, inject} from 'vue'
import {ElMessage} from 'element-plus'
import {taskModules, taskModule, taskName, taskUrl, taskStatusType} from '../task-center'
import {runningStatuses} from '../exploration'
import TaskDetail from './TaskDetail.vue'
import TaskReconfigure from './TaskReconfigure.vue'
import TableFilters from './TableFilters.vue'
import {filterRows, useTableFilters} from '../table-filters'

const props = defineProps({module:String, id:String, jobs:Array, datasets:Array, parameterCatalog:Object, catalog:Object, stopping:Array})
const emit = defineEmits(['stop','created'])
const theme = inject('platformTheme')
const editing = ref(null), page = ref(1)
const rows = computed(() => props.jobs.filter(job => taskModule(job) === props.module))
const {drafts, conditions} = useTableFilters()
const fields = [
  {key:'status', label:'状态'}, {key:'dataset_name', label:'数据集'},
  {key:'task', label:'任务', read:taskName}, {key:'version', label:'数据版本', read:job => job.dataset_revision || 1},
  {key:'date', label:'提交日期', read:job => new Date(job.created_at).toLocaleDateString()},
]
const filtered = computed(() => filterRows(rows.value, fields, conditions.value))
watch(() => props.module, () => { drafts.value = []; page.value = 1 })
watch(conditions, () => { page.value = 1 })
watch(() => filtered.value.length, count => { page.value = Math.min(page.value, Math.max(1, Math.ceil(count/20))) })
watch(() => [props.module,props.id], () => { editing.value = null })
function selectModule(module) { location.hash = taskUrl(module) }
function open(job) { location.hash = taskUrl(taskModule(job), job.id) }
function created(job) {
  editing.value = null
  emit('created', job)
  open(job)
  ElMessage.success('已提交新任务')
}
</script>
<template>
  <el-tabs :model-value="module" class="workbench-tabs" @update:model-value="selectModule"><el-tab-pane v-for="(label,key) in taskModules" :key="key" :name="key" :label="label"/></el-tabs>
  <template v-if="id">
    <a class="task-back" :href="taskUrl(module)">← 返回{{taskModules[module]}}任务列表</a>
    <TaskDetail :key="id" :id="id" :jobs="jobs" :stopping="stopping" @open="open" @edit="editing=$event" @stop="emit('stop',$event)"/>
  </template>
  <section v-else class="panel"><div class="panel-heading"><h2>历史任务</h2><span class="muted" role="status">显示 {{filtered.length}} / {{rows.length}} 个任务</span></div>
    <TableFilters :key="module" v-model="drafts" :fields="fields" :rows="rows"/>
    <el-table :data="filtered.slice((page-1)*20,page*20)" row-key="id" stripe :max-height="theme.table==='fixed'?520:undefined" :empty-text="rows.length?'没有符合条件的任务':'该模块还没有提交过任务'">
      <el-table-column label="任务" min-width="136"><template #default="{row}"><a :href="taskUrl(module,row.id)">{{taskName(row)}}</a></template></el-table-column>
      <el-table-column label="数据集" min-width="142"><template #default="{row}">{{row.dataset_name}} · v{{row.dataset_revision||1}}</template></el-table-column>
      <el-table-column label="提交时间" min-width="158"><template #default="{row}">{{new Date(row.created_at).toLocaleString()}}</template></el-table-column>
      <el-table-column label="状态" width="86"><template #default="{row}"><el-tag :type="taskStatusType(row.status)">{{row.status}}</el-tag></template></el-table-column>
      <el-table-column label="进度与结果" min-width="170"><template #default="{row}"><el-progress v-if="runningStatuses.includes(row.status)" :percentage="row.progress"/><span :title="row.message">{{row.message}}</span></template></el-table-column>
      <el-table-column label="操作" min-width="190"><template #default="{row}">
        <el-button link type="primary" @click="open(row)">{{row.status==='成功'?'查看报告':'查看详情'}}</el-button><el-button link @click="editing=row">重新配置</el-button>
        <el-button v-if="runningStatuses.includes(row.status)" link type="danger" :loading="row.status==='停止中'||stopping.includes(row.id)" @click="emit('stop',row)">停止</el-button>
      </template></el-table-column>
    </el-table>
    <el-pagination v-if="filtered.length>20" v-model:current-page="page" :page-size="20" :total="filtered.length" layout="prev,pager,next"/>
  </section>
  <TaskReconfigure v-if="editing" :key="editing.id" :job="editing" :datasets="datasets" :catalog="catalog" :parameter-catalog="parameterCatalog" @close="editing=null" @created="created"/>
</template>
<style scoped>
a{color:var(--accent-text);text-decoration:none}a:hover{text-decoration:underline}.task-back{display:inline-block;margin:0 0 14px}
</style>
