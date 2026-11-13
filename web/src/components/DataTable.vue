<script setup>
import {computed,inject,ref,watch} from 'vue'
import TableFilters from './TableFilters.vue'
import {filterRows, summaryFields, useTableFilters} from '../table-filters'
const props=defineProps({rows:{type:Array,default:()=>[]},columnGroups:{type:Array,default:()=>[]},height:{default:440},fieldFilters:Boolean})
const theme=inject('platformTheme',{table:'auto'})
const page=ref(1),search=ref('')
const {drafts,conditions}=useTableFilters()
const columns=computed(()=>props.columnGroups.length?props.columnGroups:Object.keys(props.rows[0]||{}).map(key=>({key,label:key})))
const fields=computed(()=>summaryFields(columns.value))
const filtered=computed(()=>filterRows(props.rows,fields.value,conditions.value).filter(r=>!search.value||Object.values(r).some(v=>String(v).toLowerCase().includes(search.value.toLowerCase()))))
watch(()=>props.rows,()=>{page.value=1;search.value='';drafts.value=[]})
watch([search,conditions],()=>page.value=1)
const format=v=>v===null||v===undefined?'—':typeof v==='number'&&!Number.isInteger(v)?Number(v.toPrecision(6)):typeof v==='object'?JSON.stringify(v):String(v)
</script>
<template><div class="data-table">
  <TableFilters v-if="fieldFilters" v-model="drafts" :fields="fields" :rows="rows" :has-search="!!search" @reset="search=''">
    <template #search><el-input v-model="search" placeholder="搜索表格内容" clearable aria-label="搜索表格内容"/></template>
  </TableFilters>
  <div class="table-tools"><span class="muted" role="status">{{fieldFilters ? `显示 ${filtered.length} / ${rows.length} 个字段` : `共 ${filtered.length} 条`}}</span><el-input v-if="!fieldFilters" v-model="search" placeholder="搜索表格内容" clearable aria-label="搜索表格内容" style="width:240px"/></div>
  <el-table :data="filtered.slice((page-1)*20,page*20)" :max-height="theme.table==='fixed'?height:''" stripe :empty-text="fieldFilters?'没有符合条件的字段':'暂无结果'" style="width:100%">
  <template v-for="c in columns" :key="c.key||c.label">
    <el-table-column v-if="c.children" :label="c.label" align="center"><el-table-column v-for="child in c.children" :key="child.key" :prop="child.key" :label="child.label" min-width="124" sortable show-overflow-tooltip><template #default="{row}">{{format(row[child.key])}}</template></el-table-column></el-table-column>
    <el-table-column v-else :prop="c.key" :label="c.label" min-width="124" sortable show-overflow-tooltip><template #default="{row}">{{format(row[c.key])}}</template></el-table-column>
  </template>
</el-table><el-pagination v-if="filtered.length>20" v-model:current-page="page" :total="filtered.length" :page-size="20" :pager-count="5" layout="prev,pager,next,total"/></div></template>
