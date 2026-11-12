<script setup>
import {computed} from 'vue'
import {modelDescription,modelOptions,modelParameterSpecs,selectedModelParameters} from '../model-options'
import OptionalParameters from './OptionalParameters.vue'

const props=defineProps({models:{type:Array,required:true},parameters:{type:Object,default:()=>({})},fitParameters:{type:Object,default:()=>({})},catalog:{type:Object,required:true},compare:Boolean,sharedEditor:Boolean})
const extraValues=row=>({model:Object.fromEntries(Object.entries(row.params).filter(([key])=>!row.specs.some(s=>s.key===key))),fit:props.fitParameters[row.key]||{}})
function extraGroup(row){
  const model=props.catalog[row.key]
  const groups={model:{label:'模型构造参数',params:model?.extra_params||{}}}
  if(Object.keys(model?.fit_params||{}).length)groups.fit={label:'模型拟合参数',params:model.fit_params}
  return groups
}
const emit=defineEmits(['update:models','update:parameters','update:fitParameters'])
const options=computed(()=>modelOptions(props.catalog))
const pickerProps=computed(()=>({multiple:props.compare,emitPath:false}))
const selection=computed({
  get:()=>props.compare?props.models:props.models[0],
  set:value=>emit('update:models',Array.isArray(value)?value:value?[value]:[]),
})
const rows=computed(()=>{
  const effective=selectedModelParameters(props.models,props.catalog,props.parameters)
  return props.models.map(key=>{
    const model=props.catalog[key]
    return {key,name:model?.name||key,className:model?.class_name||key,description:model?modelDescription(model):'',
      params:effective[key],json:JSON.stringify(effective[key]),specs:modelParameterSpecs(model)}
  })
})
function applyModel(key,values,extra){
  emit('update:parameters',{...props.parameters,[key]:{...values,...extra.model}})
  emit('update:fitParameters',{...props.fitParameters,[key]:extra.fit||{}})
}
</script>
<template>
  <div class="form-grid"><el-form-item :label="compare?'对比模型（可多选）':'训练模型（单选）'" required>
    <el-cascader v-model="selection" class="training-model-picker" :class="{'scrollable-tags':compare}" popper-class="training-model-menu" :options="options" :props="pickerProps"
      placeholder="选择模型类别和模型" separator=" > " filterable
      :filter-method="(node,query)=>node.text.toLowerCase().includes(query.toLowerCase())">
      <template #default="{data}"><span :title="data.reason||undefined">{{data.label}}</span></template>
    </el-cascader>
  </el-form-item></div>
  <OptionalParameters v-if="!sharedEditor&&!compare&&rows.length" :key="rows[0].key" :title="rows[0].name+'参数'" :specs="rows[0].specs" :values="rows[0].params"
    :extensions="extraGroup(rows[0])" :overrides="extraValues(rows[0])" @apply="(values,extra)=>applyModel(rows[0].key,values,extra)"/>
  <el-table v-if="compare&&rows.length" :data="rows" row-key="key" stripe aria-label="对比模型参数列表" class="comparison-models">
    <el-table-column prop="className" label="类名" min-width="150" show-overflow-tooltip/>
    <el-table-column prop="name" label="模型名" min-width="105" show-overflow-tooltip/>
    <el-table-column prop="description" label="描述" min-width="175" show-overflow-tooltip/>
    <el-table-column prop="json" label="参数" min-width="185" :show-overflow-tooltip="{popperClass:'model-params-tip'}">
      <template #default="{row}"><code class="model-json" :aria-label="row.name+'参数 JSON'">{{row.json}}</code></template>
    </el-table-column>
    <el-table-column label="操作" width="94" fixed="right"><template #default="{row}">
      <OptionalParameters :key="row.key" row-editor :title="row.name+'参数'" :specs="row.specs" :values="row.params"  :extensions="extraGroup(row)" :overrides="extraValues(row)" @apply="(values,extra)=>applyModel(row.key,values,extra)"/>

    </template></el-table-column>
  </el-table>
</template>

<style>
.model-params-tip{max-width:min(620px,90vw);overflow-wrap:anywhere;white-space:normal}
.comparison-models{margin-bottom:14px}.model-json{font-family:ui-monospace,monospace;white-space:nowrap;font-size:12px}
.training-model-picker { width: 100%; min-width: 0; }
@media (max-width: 600px) {
  .training-model-menu .el-cascader-menu { min-width: 0; max-width: calc(55vw - 10px); }
  .training-model-menu .el-cascader-menu:first-child { max-width: calc(45vw - 10px); }
  .training-model-menu .el-cascader-node { padding-inline: 10px; }
}
</style>
