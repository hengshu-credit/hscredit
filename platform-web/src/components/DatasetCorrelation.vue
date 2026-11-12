<script setup>
import {computed,ref,shallowRef,watch} from 'vue'
import {api} from '../api'
import Chart from './Chart.vue'

const props=defineProps({dataset:{type:Object,required:true},themeKey:String})
const rowPage=ref(1), columnPage=ref(1), pageSize=ref(16), retry=ref(0)
const result=shallowRef(null), error=ref('')
const total=computed(()=>props.dataset.columns.length)
const pages=computed(()=>Array.from({length:Math.ceil(total.value/pageSize.value)},(_,i)=>{
  const start=i*pageSize.value, end=Math.min(start+pageSize.value,total.value)
  return `${start+1}–${end} · ${props.dataset.columns[start].name} … ${props.dataset.columns[end-1].name}`
}))
const kinds=computed(()=>new Map(props.dataset.columns.map(c=>[c.name,c.kind])))
watch(pageSize,()=>{rowPage.value=1;columnPage.value=1})
watch([rowPage,columnPage,pageSize,retry],async ([row,column,size],_,onCleanup)=>{
  const controller=new AbortController()
  onCleanup(()=>controller.abort())
  result.value=null
  error.value=''
  try{
    const params=new URLSearchParams({revision:props.dataset.revision||1,row_page:row,column_page:column,page_size:size})
    const matrix=await api(`/datasets/${props.dataset.id}/correlation?${params}`,{signal:controller.signal})
    if(!controller.signal.aborted)result.value=matrix
  }catch(e){
    if(!controller.signal.aborted)error.value=e.message
  }
},{immediate:true})

function method(a,b){
  const count=Number(kinds.value.get(a)==='category')+Number(kinds.value.get(b)==='category')
  return ['Pearson r','相关比 η','Cramér’s V'][count]
}
const display=value=>value===null?'—':value.toFixed(4)
const chart=computed(()=>{
  const {rows,columns,values}=result.value
  const data=[]
  for(const [i,row] of values.entries()){
    row.forEach((value,j)=>{if(value!==null)data.push([j,i,value])})
  }
  return {
    animation:false,
    tooltip:{trigger:'item',formatter:({value:[j,i,v]})=>`${rows[i]} × ${columns[j]}\n${method(rows[i],columns[j])}  ${display(v)}`},
    grid:{left:8,right:16,top:10,bottom:54,containLabel:true},
    xAxis:{type:'category',data:columns,axisLabel:{rotate:35,width:104,overflow:'truncate',hideOverlap:true}},
    yAxis:{type:'category',inverse:true,data:rows,axisLabel:{width:104,overflow:'truncate',hideOverlap:true}},
    visualMap:{min:-1,max:1,orient:'horizontal',left:'center',bottom:0,text:['+1','−1'],inRange:{color:['#b45309','#f1f5f9','#1d4ed8']}},
    series:[{type:'heatmap',data}]
  }
})
</script>

<template>
  <section class="panel correlation-panel" aria-label="字段相关性">
    <div class="panel-heading"><h2>字段相关性</h2><span class="muted">{{dataset.name}} · {{total}} 个字段</span></div>
    <p class="muted">数值之间用 Pearson r，类别之间用 Cramér’s V，类别与数值之间用相关比 η。日期按时间先后、布尔值按 0/1 参与数值计算。</p>
    <div v-if="total>16" class="matrix-controls">
      <label>每页字段数 <select v-model.number="pageSize"><option v-for="size in [16,32,64]" :key="size" :value="size">{{size}}</option></select></label>
      <template v-if="pages.length>1">
        <label>行字段 <select v-model.number="rowPage"><option v-for="(name,i) in pages" :key="i" :value="i+1">{{name}}</option></select></label>
        <label>列字段 <select v-model.number="columnPage"><option v-for="(name,i) in pages" :key="i" :value="i+1">{{name}}</option></select></label>
      </template>
    </div>
    <p v-if="error" role="alert">{{error}} <el-button link type="primary" @click="retry++">重试</el-button></p>
    <template v-else-if="result">
      <h3>相关性热力图</h3>
      <Chart :option="chart" :theme-key="themeKey" aria-label="字段相关性热力图" :style="{height:Math.max(280,Math.min(780,140+result.rows.length*24))+'px'}"/>
      <h3>相关性表</h3>
      <p class="muted">每对字段使用共同有效的样本。常量或有效样本不足时，图中留空，表中显示「—」。</p>
      <div class="matrix-scroll" tabindex="0">
        <table aria-label="相关性表">
          <thead><tr><th scope="col">字段</th><th v-for="name in result.columns" :key="name" scope="col">{{name}}</th></tr></thead>
          <tbody><tr v-for="(name,i) in result.rows" :key="name"><th scope="row">{{name}}</th>
            <td v-for="(v,j) in result.values[i]" :key="j" :title="name+' × '+result.columns[j]+' · '+method(name,result.columns[j])">{{display(v)}}</td>
          </tr></tbody>
        </table>
      </div>
    </template>
    <p v-else role="status" class="muted">正在计算字段相关性…</p>
  </section>
</template>

<style scoped>
.correlation-panel{min-width:0}.correlation-panel h3{font-size:14px;margin:18px 0 6px}.correlation-panel p{margin:6px 0 12px}
.matrix-controls,.matrix-controls label{display:flex;align-items:center;flex-wrap:wrap;gap:6px}.matrix-controls{gap:8px 16px}.matrix-controls label{min-width:0;font-size:13px}
.matrix-controls select{max-width:min(300px,100%);height:30px;color:var(--text);background:var(--surface);border:1px solid var(--border)}
.matrix-scroll{max-height:420px;overflow:auto;border:1px solid var(--border)}table{width:100%;border-collapse:separate;border-spacing:0;font-size:12px;font-variant-numeric:tabular-nums}
th,td{white-space:nowrap;text-align:right;padding:6px 10px;border-bottom:1px solid var(--line-soft);border-right:1px solid var(--line-soft)}
th{background:var(--surface);text-align:left}thead th{position:sticky;top:0;background:var(--soft);z-index:1}th:first-child{position:sticky;left:0;z-index:1}thead th:first-child{z-index:2}
</style>
