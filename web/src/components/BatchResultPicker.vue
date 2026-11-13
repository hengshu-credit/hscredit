<script setup>
import {computed, inject, ref} from 'vue'
const props = defineProps({job: Object})
const emit = defineEmits(['select'])
const jobs = inject('platformJobs', ref([]))
const siblings = computed(() => props.job?.batch_id
  ? jobs.value.filter(item => item.batch_id === props.job.batch_id).sort((a,b) => a.batch_position-b.batch_position)
  : [])
</script>
<template>
  <div v-if="siblings.length" class="batch-result-picker">
    <strong>查看本次运行结果</strong>
    <el-select :model-value="job.id" aria-label="查看本次运行结果" @change="id=>emit('select',siblings.find(item=>item.id===id))">
      <el-option v-for="item in siblings" :key="item.id" :value="item.id"
        :label="item.dataset_name+' · 版本 '+item.dataset_revision+' · '+item.status"/>
    </el-select>
  </div>
</template>
<style scoped>
.batch-result-picker{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:16px 0}
.batch-result-picker .el-select{width:min(390px,100%)}
</style>
