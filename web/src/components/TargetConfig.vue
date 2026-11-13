<script setup>
import {computed,ref,watch} from 'vue'
import {Setting} from '@element-plus/icons-vue'
import TargetFields from './TargetFields.vue'
import {copyTargetDraft,useTargetConfig} from '../target-config'
const props=defineProps({config:{type:Object,required:true},dataset:{type:Object,required:true},training:{type:Boolean,default:true}})
const emit=defineEmits(['applied'])
const opened=ref(false)
const editor=useTargetConfig(()=>props.dataset,()=>props.config.purpose)
const modes={none:'无标签',target:'现有标签',overdue:'逾期天数',dict:'标签字典'}
const summary=computed(()=>{
  if(props.config.error)return props.config.error
  if(props.config.draft.mode==='none')return '无标签'
  const name=props.training&&props.config.purpose==='training'?'主标签 · '+props.config.primary:props.config.conditions[0]
  return modes[props.config.draft.mode]+' · '+name+(props.config.conditions.length>1?'（共 '+props.config.conditions.length+' 个口径）':'')
})
watch(()=>[props.dataset.id,props.dataset.revision],()=>{opened.value=false})
function edit(){copyTargetDraft(editor.draft,props.config.draft);opened.value=true}
function apply(){if(!editor.error){copyTargetDraft(props.config.draft,editor.draft);opened.value=false;emit('applied',props.config.payload)}}
</script>
<template>
  <div class="config-summary"><div><strong>目标变量</strong><small>{{config.purpose==='explore'?'可选':'必选'}}</small><span :class="{'config-invalid':config.error}">{{summary}}</span></div>
    <el-button plain :icon="Setting" @click="edit">配置目标变量</el-button></div>
  <el-dialog v-model="opened" title="配置目标变量" width="min(740px, 95vw)" append-to-body destroy-on-close :close-on-click-modal="false">
    <div class="parameter-body"><TargetFields :config="editor" :dataset="dataset" :training="training"/></div>
    <template #footer><el-button @click="opened=false">取消</el-button><el-button type="primary" :disabled="!!editor.error" @click="apply">应用配置</el-button></template>
  </el-dialog>
</template>
