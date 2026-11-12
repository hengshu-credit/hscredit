<script setup>
import FieldSelect from "./FieldSelect.vue"
import {conditionOperators,emptyCondition} from '../split-tree'
const props=defineProps({node:Object,columns:Array,depth:{type:Number,default:0}})
const emit=defineEmits(['remove'])
function setField(leaf){leaf.value=leaf.operator==='between'?['','']:['in','not_in'].includes(leaf.operator)?[]:null}
function setOperator(leaf){setField(leaf)}
function values(leaf){return (props.columns.find(c=>c.name===leaf.field)?.values||[]).map(v=>v.value).filter(v=>v!==null)}
</script>
<template>
  <div class="split-condition-group">
    <div class="split-condition-head"><strong>{{depth?'条件组':'满足以下条件'}}</strong><el-radio-group v-model="node.op" size="small" aria-label="条件关系"><el-radio-button value="AND">全部满足</el-radio-button><el-radio-button value="OR">任一满足</el-radio-button></el-radio-group><el-button v-if="depth" text type="danger" size="small" @click="emit('remove')">删除组</el-button></div>
    <div class="split-condition-stem">
      <template v-for="(child,index) in node.children" :key="index">
        <SplitCondition v-if="child.type==='group'" :node="child" :columns="columns" :depth="depth+1" @remove="node.children.splice(index,1)"/>
        <div v-else class="split-condition-leaf">
          <FieldSelect v-model="child.field" :fields="columns" label="条件字段" @change="setField(child)"/>
          <el-select v-model="child.operator" aria-label="条件运算符" @change="setOperator(child)"><el-option v-for="(label,op) in conditionOperators" :key="op" :value="op" :label="label"/></el-select>
          <div v-if="child.operator==='between'" class="split-condition-range"><el-input v-model="child.value[0]" placeholder="区间下限" aria-label="区间下限"/><span>至</span><el-input v-model="child.value[1]" placeholder="区间上限" aria-label="区间上限"/></div>
          <el-select v-else-if="['in','not_in'].includes(child.operator)" v-model="child.value" multiple filterable allow-create default-first-option placeholder="选择或输入取值后回车" aria-label="条件集合"><el-option v-for="(value,i) in values(child)" :key="i" :value="value" :label="String(value)"/></el-select>
          <el-select v-else-if="!['is_null','not_null'].includes(child.operator)" v-model="child.value" filterable allow-create default-first-option placeholder="选择或输入比较值后回车" aria-label="条件值"><el-option label="空文本" value=""/><el-option v-for="(value,i) in values(child).filter(v=>v!=='')" :key="i" :value="value" :label="String(value)"/></el-select>
          <span v-else class="muted">无需比较值</span>
          <el-button text type="danger" size="small" aria-label="删除条件" @click="node.children.splice(index,1)">删除</el-button>
        </div>
      </template>
      <div class="split-condition-tools"><el-button class="append-control" @click="node.children.push({type:'leaf',field:'',operator:'eq',value:null})">加条件</el-button><el-button class="append-control" :disabled="depth>=5" @click="node.children.push(emptyCondition())">加条件组</el-button></div>
    </div>
  </div>
</template>
