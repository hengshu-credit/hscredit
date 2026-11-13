<script setup>
import {computed, watch} from 'vue'
import TargetConfig from './TargetConfig.vue'
import {useTargetConfig} from '../target-config'
const props=defineProps({config:{type:Object,required:true},dataset:{type:Object,required:true}})
const emit=defineEmits(['apply'])
const purpose=props.config.kind==='explore'?'explore':['model','strategy'].includes(props.config.kind)?'training':'report'
const target=useTargetConfig(()=>props.dataset,purpose)
const vectors=computed(()=>props.config.target && typeof props.config.target==='object' && Object.values(props.config.target).some(Array.isArray))
watch([()=>props.config,()=>props.dataset],()=>{
  if(vectors.value)return
  const cfg=props.config,d=target.draft
  d.missing=cfg.handle_missing_target||'drop'
  d.primary=''
  if(cfg.label_mode==='overdue'){
    Object.assign(d,{mode:'overdue',overdue:[...(cfg.overdue||[])],text:(cfg.dpds||[]).join(', '),operator:cfg.overdue_operator||'>',grey:!!cfg.del_grey})
  }else if(cfg.target && typeof cfg.target==='object'){
    const names=cfg.target_order?.length?cfg.target_order:Object.keys(cfg.target)
    d.mode='dict';d.dictionaryGrey=!!cfg.del_grey
    d.entries=names.map(name=>{
      const entry=target.newEntry(),rule=cfg.target[name]
      return {...entry,name,field:rule.field,operator:rule.operator||'raw',value:rule.value??null}
    })
  }else{
    const names=purpose==='explore'?(cfg.targets||[]):[cfg.target,...(cfg.evaluation_targets||[])].filter(Boolean)
    d.mode=names.length||purpose!=='explore'?'target':'none'
    d.fields=[...names]
    d.positives=Object.fromEntries(names.map(name=>[name,cfg.positive_labels?.[name]??(name===cfg.target?(cfg.positive_label??1):1)]))
  }
  if(cfg.primary_index && d.mode!=='target' && target.conditions[cfg.primary_index])target.primary=target.conditions[cfg.primary_index]
},{immediate:true})
</script>
<template>
  <div v-if="vectors" class="config-summary"><strong>目标变量</strong><span>原任务使用逐行标签向量，可在完整配置中编辑。</span></div>
  <TargetConfig v-else :config="target" :dataset="dataset" @applied="emit('apply',$event)"/>
</template>
