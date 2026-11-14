<script setup>
import TableFrame from './TableFrame.vue'
import {computed,defineAsyncComponent,reactive,ref,shallowRef,watch} from 'vue'
import {api} from '../api'
import {rememberForm} from '../form-drafts'
import Chart from './Chart.vue'
import DataTable from './DataTable.vue'
import ExplorePivot from './ExplorePivot.vue'
import {chosenFormat,crossCounts,distributionRows,fieldChart,fieldChoices,kindNames,pairChart,pairChoices} from '../explore-formats'
import {summaryRows} from '../exploration-chart'
import {profileValue} from '../column-profile'

const Chart3D=defineAsyncComponent(()=>import('./Chart3D.vue'))
const props=defineProps({result:{type:Object,required:true},summary:Object,job:{type:Object,required:true},formats:{type:Map,required:true},pairFormats:{type:Map,required:true},themeKey:String})
const emit=defineEmits(['format','pair-format'])
const filters=reactive(new Map()),show3D=ref(true),labelled=ref(true),jointView=ref('chart')
const targetName=ref(''),population=ref('valid'),targetCounts=shallowRef(props.result.target_counts||[])
const result=shallowRef(props.result),targetError=ref(''),loadingSample=ref(false),retry=ref(0)
const appliedTarget=ref(''),appliedPopulation=ref('valid')
const defaults={width:64,top:425,bottom:305,columns:3,chartHeight:280}
const layout=reactive({...defaults})
rememberForm('explore-view/'+props.job.id,
  ()=>({layout,filters:[...filters],joint:jointView.value,show3D:show3D.value,labelled:labelled.value,target:targetName.value,population:population.value}),
  saved=>{
    Object.assign(layout,saved.layout)
    for(const [key,value] of saved.filters)filters.set(key,value)
    jointView.value=saved.joint;show3D.value=saved.show3D;labelled.value=saved.labelled
    targetName.value=saved.target;population.value=saved.population
  })
watch([targetName,population,retry],async ([name,sample],_,cleanup)=>{
  const controller=new AbortController()
  cleanup(()=>controller.abort())
  targetError.value=''
  loadingSample.value=!!name
  const path='/jobs/'+props.job.id+'/target-view'
  try{
    if(props.job.hasTargets&&!props.result.target_counts&&!targetCounts.value.length){
      const counts=await api(path,{signal:controller.signal})
      if(controller.signal.aborted)return
      targetCounts.value=counts
    }
    if(name){
      const chart=await api(path+'?'+new URLSearchParams({target:name,population:sample}),{signal:controller.signal})
      if(controller.signal.aborted)return
      result.value=chart
    }else{
      result.value=props.result
    }
    appliedTarget.value=name
    appliedPopulation.value=sample
  }catch(e){if(!controller.signal.aborted)targetError.value=e.message}
  finally{if(!controller.signal.aborted)loadingSample.value=false}
},{immediate:true})
const cards=computed(()=>(result.value?.distributions||[]).map((dist,i)=>{
  const choices=fieldChoices(dist.kind),format=chosenFormat(choices,props.formats.get(dist.feature))
  return {dist,i,choices,format,option:format==='table'?null:fieldChart(dist,format),
    stats:summaryRows(props.summary,dist.feature).map(r=>({...r,value:profileValue(r.name,r.value)}))}
}))
const pairs=computed(()=>(result.value?.pairs||[]).map(pair=>{
  const names=pair.indices.map(i=>result.value.axes[i].name)
  const key=JSON.stringify(names),choices=pairChoices(pair.indices.map(i=>result.value.axes[i].kind))
  const format=chosenFormat(choices,props.pairFormats.get(key))
  const labels=[...new Set(pair.density.map(r=>r['标签']))]
  const group=labels.includes(filters.get(key))?filters.get(key):null
  return {pair,key,names,choices,format,labels,group,option:format==='table'?null:pairChart(pair,result.value.axes,format,group)}
}))
const pivotBrief=ref(null)
const gridStyle=computed(()=>({'--main-width':layout.width+'%','--top-height':layout.top+'px','--bottom-height':layout.bottom+'px','--columns':layout.columns,'--chart-height':layout.chartHeight+'px'}))
const pairBrief=computed(()=>{
  const item=pairs.value[0]
  if(!item)return null
  const cells=crossCounts(item.pair,item.group)
  const total=cells.reduce((sum,c)=>sum+c.count,0)
  const largest=cells.reduce((best,c)=>!best||c.count>best.count?c:best,null)
  return {total,largest,occupied:cells.length}
})
const count=n=>n?.toLocaleString()??'—'
</script>

<template>
  <section class="panel exploration-result" aria-label="组合分析结果">
    <header class="result-heading"><h2>分析结果</h2><span class="muted">{{job.name}} · {{count(props.result.total)}} 行 · {{props.result.axes.length}} 个字段</span></header>
    <section v-if="targetCounts.length" class="target-results" aria-label="目标标签概况">
      <h3>目标标签概况</h3>
      <TableFrame v-slot="{viewportStyle}"><div class="target-table" :style="viewportStyle"><table><thead><tr><th v-for="key in ['标签','有效样本','好样本','坏样本','剔除样本','坏样本率']" :key="key">{{key}}</th></tr></thead>
        <tbody><tr v-for="row in targetCounts" :key="row['标签']"><th scope="row">{{row['标签']}}</th><td v-for="key in ['有效样本','好样本','坏样本','剔除样本']" :key="key">{{count(row[key])}}</td><td>{{row['坏样本率']==null?'—':(100*row['坏样本率']).toFixed(2)+'%'}}</td></tr></tbody>
      </table></div></TableFrame>
      <div class="target-controls"><label>图表目标标签 <select v-model="targetName"><option value="">全部样本</option><option v-for="row in targetCounts" :key="row['标签']">{{row['标签']}}</option></select></label>
        <label v-if="targetName">样本范围 <select v-model="population"><option value="valid">有效样本</option><option value="good">好样本</option><option value="bad">坏样本</option></select></label>
        <span v-if="targetName&&!loadingSample&&!targetError" class="muted">当前样本 {{count(result.total)}} 行</span>
      </div>
    </section>
    <p v-if="targetError" role="alert">{{targetError}} <el-button link @click="retry++">重试</el-button></p>
    <p v-if="loadingSample" role="status">正在按目标标签更新图表…</p>
    <div v-show="!loadingSample&&!(targetName&&targetError)">
    <details class="size-options"><summary>布局与显示</summary><div>
      <template v-if="result.axes.length===2">
      <label>主列宽度 {{layout.width}}%<input v-model.number="layout.width" type="range" min="40" max="76" aria-label="主列宽度"/></label>
      <label>首行高度 {{layout.top}} px<input v-model.number="layout.top" type="range" min="270" max="800" step="5" aria-label="首行高度"/></label>
      <label>次行高度 {{layout.bottom}} px<input v-model.number="layout.bottom" type="range" min="200" max="700" step="5" aria-label="次行高度"/></label>
      </template><template v-else>
        <label v-if="result.axes.length>1">每行图表数<select v-model.number="layout.columns" aria-label="每行图表数"><option v-for="n in [1,2,3]" :key="n" :value="n">{{n}} 列</option></select></label>
        <label>子图高度 {{layout.chartHeight}} px<input v-model.number="layout.chartHeight" type="range" min="200" max="700" step="5" aria-label="子图高度"/></label>
      </template>
      <el-button link @click="Object.assign(layout,defaults)">恢复默认</el-button>
    </div></details>
    <details v-if="result.axes.length===3" class="three-d" :open="show3D" @toggle="show3D=$event.target.open"><summary>三变量联合分析 · {{count(result.valid)}} 行共同有效</summary>
      <template v-if="show3D"><el-radio-group v-model="jointView" size="small" aria-label="联合视图"><el-radio-button value="chart">三维分布</el-radio-button><el-radio-button value="pivot">多级透视表</el-radio-button></el-radio-group>
        <template v-if="jointView==='chart'"><el-checkbox v-if="result.target" v-model="labelled">按 {{result.target}} 着色</el-checkbox><Chart3D :data="result" :labelled="labelled" :theme-key="themeKey"/></template>
        <KeepAlive><ExplorePivot v-if="jointView==='pivot'" :axes="result.axes" :row-fields="result.axes.slice(0,2).map(a=>a.name)" :column-fields="[result.axes[2].name]" :job="job" :target="appliedTarget" :population="appliedPopulation"/></KeepAlive>
      </template>
    </details>
    <div class="results-grid" :class="{'two-axis':result.axes.length===2,'single-axis':result.axes.length===1}" :style="gridStyle">
      <h3 v-if="pairs.length" class="section-title">二维交叉分析</h3>
        <section v-for="item in pairs" :key="item.key" class="subplot pair-panel" :aria-label="item.names.join(' × ')+'的交叉结果'">
          <header><h4>{{item.names.join(' × ')}}</h4><select :aria-label="item.names.join(' × ')+'的交叉图形'" :value="item.format" @change="emit('pair-format',item.key,$event.target.value)"><option v-for="[id,label] in item.choices" :key="id" :value="id">{{label}}</option></select></header>
          <div class="pair-info"><span v-if="item.format!=='table'" class="muted">有效配对 {{count(item.pair.valid)}}<template v-if="item.format==='scatter'"> · 全组散点 {{count(item.pair.points.length)}}</template><template v-if="item.pair.pearson!=null"> · 全组 Pearson r {{item.pair.pearson.toFixed(3)}}</template></span>
            <label v-if="result.target">{{result.target}} <select :aria-label="item.names.join(' × ')+'的分组筛选'" :value="item.group===null?-1:item.labels.indexOf(item.group)" @change="filters.set(item.key,item.labels[Number($event.target.value)]??null)"><option :value="-1">全部</option><option v-for="(name,i) in item.labels" :key="name" :value="i">{{name}}</option></select></label>
          </div>
          <Chart v-if="item.option" :option="item.option" :theme-key="themeKey"/>
          <KeepAlive><ExplorePivot v-if="item.format==='table'" :axes="result.axes" :row-fields="[item.names[1]]" :column-fields="[item.names[0]]" :job="job" :group="item.group" :target="appliedTarget" :population="appliedPopulation" @summary="pivotBrief=$event"/></KeepAlive>
          <details v-if="item.format==='means'"><summary>均值与配对样本数</summary><DataTable :rows="item.pair.means.filter(r=>item.group===null||r['标签']===item.group)" :height="260"/></details>
        </section>
      <section v-if="result.axes.length===2&&pairBrief" class="subplot brief">
        <header><h4>交叉摘要</h4></header>
        <template v-if="pairs[0].format==='table'"><dl v-if="pivotBrief">
          <dt>参与透视</dt><dd>{{count(pivotBrief.samples)}} 行</dd><dt>行列组合</dt><dd>{{count(pivotBrief.rows)}} 行 × {{count(pivotBrief.columns)}} 列</dd>
          <dt>{{pivotBrief.metric}}</dt><dd>{{pivotBrief.total}}</dd>
        </dl><p v-else class="muted">等待透视表结果</p></template>
        <dl v-else><dt>{{pairs[0].group===null?'有效配对':'所选组有效配对'}}</dt><dd>{{count(pairBrief.total)}} 行</dd>
          <dt>占当前样本</dt><dd>{{(result.total?100*pairBrief.total/result.total:0).toFixed(2)}}%</dd>
          <dt>非空交叉格</dt><dd>{{count(pairBrief.occupied)}}</dd>
          <template v-if="pairBrief.largest"><dt>最高频组合</dt><dd>{{pairBrief.largest.x}} × {{pairBrief.largest.y}}（{{count(pairBrief.largest.count)}} 行）</dd></template>
          <template v-if="pairs[0].group===null&&pairs[0].pair.pearson!=null"><dt>Pearson r</dt><dd>{{pairs[0].pair.pearson.toFixed(3)}}</dd></template>
        </dl>
      </section>
      <h3 class="section-title">字段分布</h3>
      <section v-for="card in cards" :key="card.dist.feature" class="subplot marginal" :class="card.i===0?'x-marginal':'y-marginal'" :aria-label="card.dist.feature+'的分析结果'">
        <header><h4>{{card.dist.feature}}</h4><select :aria-label="card.dist.feature+'的结果图形'" :value="card.format" @change="emit('format',card.dist.feature,$event.target.value)"><option v-for="[id,label] in card.choices" :key="id" :value="id">{{label}}</option></select></header>
        <p class="muted">{{kindNames[card.dist.kind]}} · 有效 {{count(card.dist.valid)}} · 缺失 {{count(card.dist.missing)}} · 唯一值 {{count(card.dist.unique)}}<template v-if="card.dist.invalid"> · 无效 {{count(card.dist.invalid)}}</template></p>
        <Chart v-if="card.option" :option="card.option" :theme-key="themeKey"/>
        <DataTable v-else :rows="distributionRows(card.dist)" :height="270"/>
        <p v-if="card.format==='box'" class="muted">须线为最小值和最大值，箱体显示上下四分位数及中位数。</p>
        <p v-else-if="card.dist.kind==='date'" class="muted">{{card.dist.distribution_note}}</p>
        <p v-else-if="card.dist.kind==='text'" class="muted">按非缺失文本的字符数统计。</p>
        <details v-if="card.stats.length"><summary>字段统计{{targetName?'（完整分析样本）':''}}</summary><TableFrame v-slot="{viewportStyle}"><div class="stats-scroll" :style="viewportStyle"><table><thead><tr><th>统计指标</th><th>统计值</th></tr></thead><tbody><tr v-for="r in card.stats" :key="JSON.stringify([r.group,r.name])"><td>{{r.group?r.group+' · ':''}}{{r.name}}</td><td>{{r.value}}</td></tr></tbody></table></div></TableFrame></details>
        <details v-if="card.dist.by_group?.length"><summary>按 {{result.target}} 汇总</summary><DataTable :rows="card.dist.by_group" :height="260"/></details>
      </section>
    </div>
    <p v-if="pairs.length" class="muted result-note">{{result.note}}</p>
    </div>
  </section>
</template>

<style scoped>
.exploration-result{container-type:inline-size}
.stats-scroll{overflow:auto}
.target-results h3{font-size:14px;margin:14px 0 6px}.target-table{overflow:auto;max-height:230px}.target-table table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}.target-table :is(th,td){padding:5px 12px;text-align:right;border-bottom:1px solid var(--line-soft)}.target-table th:first-child{text-align:left}.target-controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0;font-size:13px}
.results-grid{display:grid;gap:10px;grid-template-columns:repeat(var(--columns,3),minmax(0,1fr))}
.results-grid>.section-title{grid-column:1/-1}
.single-axis{grid-template-columns:minmax(0,1fr)}
.two-axis{grid-template-columns:minmax(0,var(--main-width)) minmax(0,1fr);grid-template-rows:var(--top-height) var(--bottom-height);margin-top:10px}
.two-axis>.section-title{display:none}
.two-axis>.subplot{min-height:0;overflow:auto}
.two-axis>.pair-panel{grid-area:1/1}.two-axis>.y-marginal{grid-area:1/2}.two-axis>.x-marginal{grid-area:2/1}.two-axis>.brief{grid-area:2/2}
.two-axis .pair-panel :deep(.chart),.two-axis .y-marginal :deep(.chart){height:calc(var(--top-height) - 116px)}
.two-axis .x-marginal :deep(.chart){height:calc(var(--bottom-height) - 100px)}
.brief dl{display:grid;grid-template-columns:auto 1fr;gap:12px;margin:14px 0;font-size:13px}.brief dt{color:var(--muted)}.brief dd{margin:0;overflow-wrap:anywhere}
.size-options{margin:10px 0}.size-options div{display:flex;align-items:center;flex-wrap:wrap;gap:18px}.size-options label{display:grid;gap:4px;font-size:12px}
@container(max-width:720px){.results-grid,.two-axis{grid-template-columns:minmax(0,1fr);grid-template-rows:auto}.two-axis>.subplot{grid-area:auto}.two-axis .subplot :deep(.chart){height:245px}}

.result-heading,.subplot header,.pair-info{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}.result-heading h2,.subplot h4{margin:0}.section-title{font-size:14px;margin:18px 0 9px}

.subplot{border:1px solid var(--border);padding:12px;min-width:0}.subplot h4{font-size:14px;overflow-wrap:anywhere}.subplot p{margin:5px 0}.subplot select{border:1px solid var(--border);background:var(--surface);color:var(--text);font:inherit;font-size:12px;padding:4px;max-width:100%}.subplot .chart{height:var(--chart-height,280px)}
.subplot details,.three-d{border-top:1px solid var(--line-soft);padding-top:8px;margin-top:8px}.subplot summary,.three-d summary{font-size:13px;cursor:pointer}.subplot table{border-collapse:collapse;width:100%;font-size:12px}.subplot td,.subplot th{text-align:left;border-bottom:1px solid var(--line-soft);padding:5px;overflow-wrap:anywhere}.pair-info{margin:5px 0;font-size:12px}.result-note{margin:10px 0 0}
</style>
