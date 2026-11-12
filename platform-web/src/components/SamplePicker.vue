<script setup>
import {computed} from 'vue'
const props=defineProps({tree:Array, modelValue:[Array,String], label:String, placeholder:String, multiple:Boolean, excluded:{type:Array,default:()=>[]}, limit:Number})
const emit=defineEmits(['update:modelValue'])
const options=computed(()=>({value:'id',label:'name',children:'children',disabled:node=>node.isGroup||props.excluded.includes(node.id)}))
</script>
<template>
  <el-tree-select :class="{'scrollable-tags':multiple}" :model-value="modelValue" :data="tree" :props="options" node-key="id" check-strictly
    :multiple="multiple" :show-checkbox="multiple" :multiple-limit="limit||0"
    filterable clearable default-expand-all :aria-label="label" :placeholder="placeholder||label"
    @update:model-value="value=>emit('update:modelValue',value || (multiple?[]:''))">
    <template #default="{data}"><span class="sample-option"><span>{{data.name}}</span><small v-if="!data.isGroup">{{data.rows.toLocaleString()}} 行 · v{{data.revision||1}}</small></span></template>
  </el-tree-select>
</template>
<style scoped>
.sample-option{display:flex;align-items:baseline;justify-content:space-between;gap:12px;min-width:0;flex:1;padding-right:8px}
.sample-option small{color:var(--muted);font-size:11px;white-space:nowrap}
</style>
