<script setup>
import {computed,ref,watch} from 'vue'
import {api} from '../api'
import TableFrame from './TableFrame.vue'

const props=defineProps({axes:Array,rowFields:Array,columnFields:Array,job:Object,group:String,target:String,population:String})
const emit=defineEmits(['summary'])
const rows=ref([...props.rowFields]),cols=ref([...props.columnFields]),value=ref(''),method=ref('rows'),buckets=ref(true)
const page=ref(1),columnPage=ref(1),nested=ref(true),totals=ref(true),rowHeight=ref(28),cellWidth=ref(94),digits=ref('auto')
const retry=ref(0),result=ref(null),busy=ref(false),error=ref('')
const tableHeight=ref(null)
const methods={rows:'样本数',count:'有效计数',nunique:'去重计数',sum:'求和',mean:'平均值',median:'中位数',min:'最小值',max:'最大值'}
const fields=computed(()=>props.axes.filter(axis=>props.rowFields.includes(axis.name)||props.columnFields.includes(axis.name)))
const valueFields=computed(()=>fields.value.filter(c=>['rows','count','nunique'].includes(method.value)||c.kind==='numeric'))
function swapFields(previous,next){
  const swap=name=>name===previous?next:name===next?previous:name
  rows.value=rows.value.map(swap)
  cols.value=cols.value.map(swap)
}
function moveField(name,fromRows){
  if(fromRows){
    rows.value=rows.value.filter(field=>field!==name)
    cols.value=[...cols.value,name]
  }else{
    cols.value=cols.value.filter(field=>field!==name)
    rows.value=[...rows.value,name]
  }
}
function changeMethod(next){
  method.value=next
  if(next==='rows'||!valueFields.value.some(c=>c.name===value.value))value.value=''
}
function changeValue(next){
  value.value=next||''
  if(!next)method.value='rows'
  else if(method.value==='rows')method.value=fields.value.find(c=>c.name===next)?.kind==='numeric'?'mean':'count'
}
const rowDepth=ref(0),colDepth=ref(0)
watch(rows,names=>{if(rowDepth.value>=names.length)rowDepth.value=0})
watch(cols,names=>{if(colDepth.value>=names.length)colDepth.value=0})
const shownRows=computed(()=>rows.value.slice(0,rowDepth.value||undefined))
const shownCols=computed(()=>cols.value.slice(0,colDepth.value||undefined))
function transpose(){
  const previous=rows.value
  rows.value=cols.value;cols.value=previous
  const depth=rowDepth.value
  rowDepth.value=colDepth.value;colDepth.value=depth
}
const spec=computed(()=>({rows:shownRows.value,columns:shownCols.value,value:value.value||null,aggregation:method.value,buckets:buckets.value,group:props.group??null,target:props.target||null,population:props.population||'valid'}))
const ready=computed(()=>rows.value.length&&cols.value.length&&(method.value==='rows'||value.value))
watch(spec,()=>{page.value=1;columnPage.value=1},{flush:'sync'})
watch([spec,page,columnPage,retry],async ([settings,r,c],previous,cleanup)=>{
  const controller=new AbortController()
  cleanup(()=>controller.abort())
  error.value='';busy.value=!!ready.value
  if(settings!==previous[0])result.value=null
  if(!ready.value){result.value=null;return}
  try{
    const data=await api('/jobs/'+props.job.id+'/pivot',{method:'POST',body:JSON.stringify({...settings,page:r,column_page:c}),signal:controller.signal})
    if(!controller.signal.aborted)result.value=data
  }catch(e){if(!controller.signal.aborted){result.value=null;error.value=e.message}}
  finally{if(!controller.signal.aborted)busy.value=false}
},{immediate:true})
const label=v=>v??'（缺失）'
const flat=tuple=>tuple.map(label).join(' / ')
const formatter=computed(()=>new Intl.NumberFormat('zh-CN',digits.value==='auto'?{maximumSignificantDigits:10}:{minimumFractionDigits:Number(digits.value),maximumFractionDigits:Number(digits.value)}))
const display=v=>v==null?'—':typeof v==='number'?formatter.value.format(v):v
watch(()=>result.value&&{samples:result.value.samples,rows:result.value.row_count,columns:result.value.column_count,
  total:display(result.value.total),metric:[value.value,methods[method.value]].filter(Boolean).join(' · ')},info=>emit('summary',info))
function span(keys,i,level){
  const same=j=>j>=0&&j<keys.length&&keys[i].slice(0,level+1).every((v,n)=>v===keys[j][n])
  if(same(i-1))return 0
  let end=i+1
  while(same(end))end++
  return end-i
}
</script>
<template>
  <div class="pivot">
    <div class="pivot-controls">
      <fieldset v-for="[side,names] in [['行',rows],['列',cols]]" :key="side"><legend>{{side}}字段</legend>
        <div v-for="(name,i) in names" :key="i" class="pivot-slot">
          <select :aria-label="side+'字段第'+(i+1)+'层'" :value="name" @change="swapFields(name,$event.target.value)">
            <option v-for="axis in fields" :key="axis.name" :value="axis.name">{{axis.name}}</option>
          </select>
          <el-button v-if="names.length>1" link :aria-label="'将'+name+'移至'+(side==='行'?'列':'行')" @click="moveField(name,side==='行')">{{side==='行'?'移至列':'移至行'}}</el-button>
        </div>
      </fieldset>
      <el-button link type="primary" @click="transpose">互换行列</el-button>
      <label>值字段<select :value="value" aria-label="值字段" @change="changeValue($event.target.value)"><option value="">{{method==='rows'?'按样本计数':'选择值字段'}}</option><option v-for="axis in valueFields" :key="axis.name" :value="axis.name">{{axis.name}}</option></select></label>
      <label>聚合方式<select :value="method" aria-label="聚合方式" @change="changeMethod($event.target.value)"><option v-for="(name,key) in methods" :key="key" :value="key" :disabled="!['rows','count','nunique'].includes(key)&&!fields.some(axis=>axis.kind==='numeric')">{{name}}</option></select></label>
    </div>
    <details class="pivot-settings"><summary>透视表设置</summary><div>
      <label>维度取值 <select v-model="buckets" aria-label="维度取值"><option :value="true">沿用分析轴分箱</option><option :value="false">原始取值</option></select></label>
      <label>表头 <select v-model="nested" aria-label="表头样式"><option :value="true">多级合并</option><option :value="false">平铺路径</option></select></label>
      <label>小数位 <select v-model="digits" aria-label="小数位"><option value="auto">自动</option><option v-for="d in [0,1,2,3,4,6]" :key="d" :value="String(d)">{{d}}</option></select></label>
      <label><input v-model="totals" type="checkbox"/>显示合计</label><label>行高 {{rowHeight}} px <input v-model.number="rowHeight" aria-label="透视行高" type="range" min="22" max="64"/></label>
      <label>列宽 {{cellWidth}} px <input v-model.number="cellWidth" aria-label="透视列宽" type="range" min="60" max="240"/></label>
      <label v-if="rows.length>1">行展开至 <select v-model.number="rowDepth" aria-label="行展开至"><option :value="0">全部层级</option><option v-for="n in rows.length-1" :key="n" :value="n">第 {{n}} 层</option></select></label>
      <label v-if="cols.length>1">列展开至 <select v-model.number="colDepth" aria-label="列展开至"><option :value="0">全部层级</option><option v-for="n in cols.length-1" :key="n" :value="n">第 {{n}} 层</option></select></label>
    </div></details>
    <p v-if="error" role="alert">{{error}} <el-button link @click="retry++">重试</el-button></p>
    <p v-if="busy&&!result" role="status">正在计算透视表…</p>
    <p v-else-if="!ready" role="status">请选择行字段、列字段和当前聚合需要的值字段。</p>
    <template v-else-if="result">
      <p class="muted">{{shownRows.join(' / ')}} ＼ {{shownCols.join(' / ')}} · {{value}} {{methods[method]}} · {{result.samples.toLocaleString()}} 行样本</p>
      <TableFrame v-if="result.samples" v-model="tableHeight" v-slot="{viewportStyle}">
      <div class="pivot-scroll" :style="viewportStyle" :aria-busy="busy" tabindex="0">
        <table aria-label="字段交叉透视表" :style="{'--cell-width':cellWidth+'px','--cell-height':rowHeight+'px'}">
          <thead><tr v-for="level in (nested?shownCols.length:1)" :key="level">
            <th v-if="level===1" :rowspan="nested?shownCols.length:1" :colspan="nested?shownRows.length:1">{{shownRows.join(' / ')}} ＼ {{shownCols.join(' / ')}}</th>
            <template v-for="(key,j) in result.column_keys" :key="j"><th v-if="!nested||span(result.column_keys,j,level-1)" :colspan="nested?span(result.column_keys,j,level-1):1" scope="col">{{nested?label(key[level-1]):flat(key)}}</th></template>
            <th v-if="totals&&level===1" :rowspan="nested?shownCols.length:1">行合计</th>
          </tr></thead>
          <tbody><tr v-for="(key,i) in result.row_keys" :key="i">
            <template v-if="nested"><template v-for="(v,k) in key" :key="k"><th v-if="span(result.row_keys,i,k)" :rowspan="span(result.row_keys,i,k)" scope="row">{{label(v)}}</th></template></template><th v-else scope="row">{{flat(key)}}</th>
            <td v-for="(v,j) in result.values[i]" :key="j">{{display(v)}}</td><td v-if="totals">{{display(result.row_totals[i])}}</td>
          </tr></tbody>
          <tfoot v-if="totals"><tr><th :colspan="nested?shownRows.length:1">列合计</th><td v-for="(v,j) in result.column_totals" :key="j">{{display(v)}}</td><td>{{display(result.total)}}</td></tr></tfoot>
        </table>
      </div></TableFrame><p v-else class="muted">没有有效配对样本</p>
      <div class="pivot-pages" v-if="result.row_count>30||result.column_count>12">
        <label v-if="result.row_count>30">行页 <el-pagination v-model:current-page="page" :total="result.row_count" :page-size="30" small layout="prev,pager,next"/></label>
        <label v-if="result.column_count>12">列页 <el-pagination v-model:current-page="columnPage" :total="result.column_count" :page-size="12" small layout="prev,pager,next"/></label>
      </div>
    </template>
  </div>
</template>
<style scoped>
.pivot-controls{display:flex;flex-wrap:wrap;align-items:end;gap:6px 10px;margin:8px 0}.pivot-controls label{display:grid;flex:1 1 110px;align-content:start;gap:3px;font-size:12px;min-width:0}.pivot-controls select{min-height:30px}
.pivot-controls fieldset{border:0;padding:0;margin:0;min-width:0;flex:1 1 160px}.pivot-controls legend{font-size:12px;padding:0;margin-bottom:3px}.pivot-slot{display:flex;align-items:center;gap:4px}.pivot-slot+.pivot-slot{margin-top:3px}.pivot-slot select{min-width:0;flex:1}
.pivot-settings{margin:6px 0}.pivot-settings div,.pivot-pages,.pivot-pages label{display:flex;align-items:center;flex-wrap:wrap;gap:6px 14px}.pivot-settings label,.pivot-pages{font-size:12px}.pivot-pages :deep(.el-pagination){margin:0}
.pivot-scroll{overflow:auto;max-height:440px;border:1px solid var(--border)}.pivot table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
.pivot th,.pivot td{padding:8px 12px;text-align:right;white-space:nowrap;border-right:1px solid var(--line-soft);border-bottom:1px solid var(--line-soft)}.pivot :is(th,td){min-width:var(--cell-width);height:var(--cell-height);padding:3px 8px;box-sizing:border-box}
.pivot th{font-weight:600;background:var(--surface);text-align:left}.pivot thead{position:sticky;top:0;z-index:1}.pivot thead th,.pivot tfoot td,.pivot tfoot th{background:var(--soft)}
</style>
