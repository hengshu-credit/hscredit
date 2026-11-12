<script setup>
import {onMounted,onBeforeUnmount,watch,ref} from 'vue'
import * as echarts from 'echarts'
import 'echarts-gl'
import {mix} from '../theme'
const props=defineProps({data:Object,labelled:Boolean,themeKey:String})
const root=ref(null);let chart,resize
function draw(){if(!chart||!props.data)return;const css=getComputedStyle(document.documentElement),text=css.getPropertyValue('--text').trim(),border=css.getPropertyValue('--border').trim(),primary=css.getPropertyValue('--chart-accent').trim(),secondary=css.getPropertyValue('--chart-secondary').trim();const axes=props.data.axes.map(a=>({name:a.unit?a.name+"（"+a.unit+"）":a.name,type:a.categories?'category':'value',...(a.categories?{data:a.categories}:{}),axisLine:{lineStyle:{color:border}},axisLabel:{color:text,...(a.kind==='date'?{formatter:v=>new Date(v).toISOString().slice(0,10)}:{})},nameTextStyle:{color:text},scale:true}));const labels=props.labelled?[...new Set(props.data.points.map(p=>p.label))]:['全部样本'];chart.setOption({backgroundColor:'transparent',color:[primary,secondary,mix(primary,text,.35),mix(secondary,text,.45),mix(primary,secondary,.5),mix(primary,border,.5)],tooltip:{renderMode:'richText'},legend:{textStyle:{color:text}},xAxis3D:axes[0],yAxis3D:axes[1],zAxis3D:axes[2],grid3D:{viewControl:{projection:'perspective'},axisPointer:{show:true}},series:labels.map(label=>({type:'scatter3D',name:label,symbolSize:5,data:props.data.points.filter(p=>!props.labelled||p.label===label).map(p=>p.values)}))},true)}
onMounted(()=>{chart=echarts.init(root.value);resize=new ResizeObserver(()=>chart.resize());resize.observe(root.value);draw();window.addEventListener('hscredit-theme-change',draw)})
watch(()=>[props.data,props.labelled,props.themeKey],draw,{deep:true})
onBeforeUnmount(()=>{resize?.disconnect();chart?.dispose();window.removeEventListener('hscredit-theme-change',draw)})
</script>
<template><div ref="root" class="chart" style="height:520px" role="img" aria-label="可旋转三维变量分布"/></template>
