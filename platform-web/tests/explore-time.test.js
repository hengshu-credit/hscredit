import test from 'node:test'
import assert from 'node:assert/strict'
import {ref} from 'vue'
import {useExploration} from '../src/exploration.js'
import {distributionOption,summaryRows} from '../src/exploration-chart.js'

test('时间分布按服务端顺序画折线，坐标轴标签与滑块空间分离',()=>{
  const option=distributionOption({kind:'date',distribution:[{'区间':'2025-01','样本数':2},{'区间':'2025-02','样本数':0}]})
  assert.deepEqual(option.series[0].data,[2,0])
  assert.equal(option.series[0].type,'line')
  assert.equal(distributionOption({kind:'numeric'}).series[0].type,'bar')
  const slider=option.dataZoom[1]
  const labelHeight=option.xAxis.axisLabel.width*Math.sin(option.xAxis.axisLabel.rotate*Math.PI/180)+20+option.xAxis.axisLabel.margin
  assert.ok(option.grid.bottom-labelHeight>slider.bottom+slider.height)
})

test('字段统计完整读取已计算的 summary 行，保留时间、缺失值与原始精度',()=>{
  const row={'特征名':'日期','字段类型':'datetime','样本数':1234567,'最小值':'2023-12-15 09:10:02','平均值':52.466612345,'KS':null,'分布占比':{'一月':.3,'二月':.7}}
  const result={summary:[{'特征名':'其他','样本数':9},row],summary_columns:Object.keys(row).map(key=>({key,label:key}))}
  const rows=summaryRows(result,'日期')
  assert.deepEqual(Object.fromEntries(rows.map(r=>[r.name,r.value])),Object.fromEntries(Object.entries(row).filter(([key])=>key!=='特征名')))
  assert.equal(rows.find(r=>r.name==='分布占比').value,row['分布占比'])
  assert.deepEqual(summaryRows(result,'标签'),[])
})

test('多标签 summary 按原有分组展示，不混淆两个标签的指标',()=>{
  const result={summary:[{'统计详情 / 特征名':'评分','统计详情 / 缺失率':.1,'标签A / KS':.25,'标签B / KS':.4}],summary_columns:[{label:'统计详情',children:[{key:'统计详情 / 特征名',label:'特征名'},{key:'统计详情 / 缺失率',label:'缺失率'}]},{label:'标签A',children:[{key:'标签A / KS',label:'KS'}]},{label:'标签B',children:[{key:'标签B / KS',label:'KS'}]}]}
  assert.deepEqual(summaryRows(result,'评分'),[{group:'统计详情',name:'缺失率',value:.1},{group:'标签A',name:'KS',value:.25},{group:'标签B',name:'KS',value:.4}])
})

test('修改汇总频率仅使图表失效，并提交新的频率而不复用旧任务',async()=>{
  const config=ref({dataset_id:'source',dataset_revision:1,feature:'日期',bins:20,freq:'M',targets:[],positive_labels:{}})
  const requests=[],stored=new Map()
  const api=async(path,options)=>{
    if(path==='/jobs'){
      const body=JSON.parse(options.body),id=String(requests.length+1)
      requests.push(body);stored.set(id,{config:body,result:{exploration:{feature:'日期',freq:body.freq}}});return {id}
    }
    return stored.get(path.split('/').at(-1))
  }
  const state=useExploration({config,api,refresh:async()=>{},onError:message=>assert.fail(message),clearError:()=>{}})
  await state.load();await state.settle([{id:'1',status:'成功'}])
  assert.equal(state.chartsCurrent.value,true)
  config.value.freq='D'
  assert.equal(state.chartsCurrent.value,false)
  assert.equal(state.summaryCurrent.value,true)
  await state.load();await state.settle([{id:'2',status:'成功'}])
  assert.deepEqual(requests.map(r=>r.freq),['M','D'])
  assert.equal(state.eda.value.freq,'D')
  assert.equal(state.chartsCurrent.value,true)
})
