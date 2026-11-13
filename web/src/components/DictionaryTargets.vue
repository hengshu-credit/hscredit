<script setup>
import FieldSelect from "./FieldSelect.vue"
const props = defineProps({config: {type: Object, required: true}})
const field = row => props.config.columns.find(c => c.name === row.field)
const numeric = row => field(row)?.kind === 'numeric' && !/bool/i.test(field(row).dtype)
function selectField(row, name) {
  if (!row.name.trim() || row.name === row.field) row.name = name
  row.field = name
  row.operator = numeric(row) || /bool/i.test(field(row)?.dtype) ? 'raw' : '=='
  row.value = null
}
function moveFirst(index) {
  const entries = props.config.draft.entries
  entries.unshift(...entries.splice(index, 1))
}
</script>

<template>
  <div class="dictionary-targets"><small>每个标签都需填写名称并选择来源字段；使用比较条件时还需填写比较值。</small>
    <div v-for="(row,i) in config.draft.entries" :key="row.id" class="dictionary-row">
      <label>标签名称<el-input v-model="row.name" :aria-label="'第'+(i+1)+'个标签名称'" placeholder="例如 首逾"/></label>
      <label>来源字段<FieldSelect :model-value="row.field" :fields="config.columns" label="字典标签来源字段" @update:model-value="selectField(row,$event)"/></label>
      <label>坏样本条件<el-select v-model="row.operator" aria-label="字典标签坏样本条件" @change="row.value=null">
        <el-option label="使用字段原值（1 为坏）" value="raw"/>
        <el-option v-for="op in numeric(row)?['==','!=','>','>=','<','<=']:['==','!=']" :key="op" :value="op"/>
      </el-select></label>
      <label v-if="row.operator!=='raw'">比较值
        <el-input-number v-if="numeric(row)" v-model="row.value" :controls="false" aria-label="比较值"/>
        <el-select v-else v-model="row.value" filterable allow-create default-first-option aria-label="比较值">
          <el-option v-for="item in (field(row)?.values||[]).filter(item=>item.value!=null)" :key="item.value" :value="item.value" :label="String(item.value)"/>
        </el-select>
      </label>
      <div class="row-actions">
        <el-button v-if="i>0" text @click="moveFirst(i)">移至首项</el-button>
        <el-button text @click="config.draft.entries.splice(i,1)">移除</el-button>
      </div>
    </div>
    <el-button plain @click="config.draft.entries.push(config.newEntry())">添加标签</el-button>
  </div>
</template>

<style scoped>
.dictionary-targets{display:grid;gap:10px;margin-bottom:14px}
.dictionary-targets>button{justify-self:start}
.dictionary-row{display:flex;flex-wrap:wrap;align-items:end;gap:10px 14px;padding:10px 0;border-bottom:1px solid var(--border)}
.dictionary-row label{display:grid;flex:1 1 155px;min-width:0;gap:5px;font-size:14px}
.dictionary-row .el-input-number{width:100%}
.row-actions{display:flex;align-items:center}
</style>
