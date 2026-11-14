<script setup>
import {computed, ref} from 'vue'
import {Rank, Plus} from '@element-plus/icons-vue'
import {categoryGroupsSchema} from '../editor-context'
import {usesCategoryGroups} from '../tree-edits'
import CodeEditor from './CodeEditor.vue'
import FieldSelect from './FieldSelect.vue'

const props = defineProps({mode: String, fields: Array, errors: Array})
const rows = defineModel({type: Array, required: true})
const dragged = ref(null), over = ref(null)
const groupSchemas = computed(() => new Map(props.fields
  .filter(field => usesCategoryGroups(props.mode, field))
  .map(field => [field.name, categoryGroupsSchema(field)])))

function add(action) {
  const id = rows.value.reduce((highest,row) => Math.max(highest,row.id), 0) + 1
  rows.value.push({id, action, node: 0, feature: '', values: '', categories: ''})
}
function move(from, to) {
  if (from < 0 || to < 0 || to >= rows.value.length || from === to) return
  const [row] = rows.value.splice(from, 1)
  rows.value.splice(to, 0, row)
}
function startDrag(event, row) {
  dragged.value = row.id
  event.dataTransfer.effectAllowed = 'move'
  event.dataTransfer.setData('text/plain', String(row.id))
}
function endDrag() {
  dragged.value = null
  over.value = null
}
function dragOver(event, id) {
  if (dragged.value === null) return
  event.preventDefault()
  over.value = id
}
function drop(index) {
  move(rows.value.findIndex(row => row.id === dragged.value), index)
  endDrag()
}
</script>

<template>
  <el-collapse><el-collapse-item title="手工调整节点分裂与剪枝">
    <p class="muted">留空时使用自动训练的树。调整从上到下执行，节点编号以执行该行时的树规则表为准。</p>
    <ol class="tree-edits">
      <li v-for="(row, index) in rows" :key="row.id" :class="{target: over === row.id && dragged !== row.id}"
          @dragover="dragOver($event, row.id)"
          @drop.prevent="drop(index)">
        <button type="button" class="handle" draggable="true" :aria-label="`调整第 ${index + 1} 行顺序`"
                title="拖拽排序，或按上下方向键移动" @dragstart="startDrag($event, row)" @dragend="endDrag"
                @keydown.up.prevent="move(index, index - 1)" @keydown.down.prevent="move(index, index + 1)">
          <el-icon><Rank/></el-icon>{{ index + 1 }}
        </button>
        <span class="action">{{ row.action === 'split' ? '分裂' : '剪枝' }}</span>
        <label>节点编号<el-input-number v-model="row.node" aria-label="节点编号" :min="0" :precision="0" :controls="false"/></label>
        <template v-if="row.action === 'split'">
          <div class="field"><span>分裂字段</span><FieldSelect v-model="row.feature" :fields="fields" label="分裂字段"/></div>
          <div v-if="groupSchemas.has(row.feature)" class="condition">
            <span>类别组</span><CodeEditor v-model="row.categories" :schema="groupSchemas.get(row.feature)"
                label="类别组" :rows="1" placeholder='[["A"],["B","C"]]'/>
          </div>
          <label v-else class="condition">{{ mode === 'binary' ? '分裂阈值' : '分裂阈值（逗号分隔）' }}
            <el-input v-model="row.values" aria-label="分裂阈值" :placeholder="mode === 'binary' ? '例如 35' : '例如 28, 36'"/>
          </label>
        </template>
        <el-button class="remove" link type="danger" :aria-label="`删除第 ${index + 1} 行`" @click="rows.splice(index, 1)">删除</el-button>
        <span v-if="errors[index]" class="row-error">{{ errors[index] }}</span>
      </li>
    </ol>
    <el-button :icon="Plus" @click="add('split')">添加分裂</el-button>
    <el-button :icon="Plus" @click="add('prune')">添加剪枝</el-button>
  </el-collapse-item></el-collapse>
</template>

<style scoped>
.tree-edits{list-style:none;padding:0;margin:0}
.tree-edits li{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 12px;margin-bottom:8px;border:1px solid var(--border);border-radius:4px}
.tree-edits li.target{outline:2px solid var(--accent-text);outline-offset:-2px}
.handle{display:inline-flex;align-items:center;gap:4px;cursor:grab;color:var(--muted);background:transparent;border:0;padding:5px}
.handle:active{cursor:grabbing}
.action{font-weight:600}
.tree-edits label,.field,.condition{display:grid;gap:3px;min-width:0}
.tree-edits .el-input-number{width:78px}
.field,.condition{flex:1 1 130px}
.remove{margin-left:auto}
.row-error{flex-basis:100%;color:var(--el-color-danger);font-size:12px}
</style>
