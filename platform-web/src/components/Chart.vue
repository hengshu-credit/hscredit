<script setup>
import {onMounted,onBeforeUnmount,ref,watch} from 'vue'
import * as echarts from 'echarts/core'
import {BarChart,LineChart,ScatterChart,HeatmapChart,PieChart,BoxplotChart} from 'echarts/charts'
import {GridComponent,TooltipComponent,LegendComponent,DataZoomComponent,VisualMapComponent,ToolboxComponent} from 'echarts/components'
import {CanvasRenderer} from 'echarts/renderers'
import {mix} from '../theme'
echarts.use([BarChart,LineChart,ScatterChart,HeatmapChart,PieChart,BoxplotChart,GridComponent,TooltipComponent,LegendComponent,DataZoomComponent,VisualMapComponent,ToolboxComponent,CanvasRenderer])
const props=defineProps({option:Object,themeKey:String})
const root=ref();let chart,resize
function draw(){
  if(!chart)return
  const css=getComputedStyle(document.documentElement), token=name=>css.getPropertyValue(name).trim()
  const primary=token('--chart-accent'),secondary=token('--chart-secondary'),surface=token('--surface'),text=token('--text'),muted=token('--muted'),border=token('--border')
  const option=props.option||{}
  const axis=value=>value?{...value,axisLine:{...value.axisLine,lineStyle:{...value.axisLine?.lineStyle,color:border}},axisTick:{...value.axisTick,lineStyle:{...value.axisTick?.lineStyle,color:border}},axisLabel:{...value.axisLabel,color:muted},nameTextStyle:{...value.nameTextStyle,color:muted},splitLine:{...value.splitLine,lineStyle:{...value.splitLine?.lineStyle,color:border}}}:undefined
  chart.setOption({
    ...option,backgroundColor:'transparent',color:[primary,mix(primary,text,.35),mix(secondary,surface,.25),secondary,mix(primary,surface,.4),mix(secondary,text,.45)],
    textStyle:{fontFamily:token('--app-font'),color:text},
    tooltip:{trigger:'axis',confine:true,renderMode:'richText',...option.tooltip,backgroundColor:surface,borderColor:border,textStyle:{color:text}},
    ...(option.legend?{legend:{...option.legend,textStyle:{color:muted},inactiveColor:border}}:{}),
    grid:{left:65,right:30,top:35,bottom:70,...option.grid},
    xAxis:axis(option.xAxis),yAxis:axis(option.yAxis),
    ...(option.visualMap?{visualMap:{...option.visualMap,textStyle:{color:muted},inRange:{color:[mix(primary,text,.35),surface,secondary],...option.visualMap.inRange}}}:{}),
    ...(option.dataZoom?{dataZoom:option.dataZoom.map(zoom=>({...zoom,borderColor:border,backgroundColor:surface,fillerColor:mix(primary,surface,.8),dataBackground:{lineStyle:{color:primary},areaStyle:{color:primary}},textStyle:{color:muted},handleStyle:{color:primary,borderColor:primary},moveHandleStyle:{color:primary}}))}:{}),
  },true)
}
watch(()=>[props.option,props.themeKey],draw,{deep:true,flush:'post'})
onMounted(()=>{chart=echarts.init(root.value);resize=new ResizeObserver(()=>chart.resize());resize.observe(root.value);window.addEventListener('hscredit-theme-change',draw);draw()})
onBeforeUnmount(()=>{resize?.disconnect();window.removeEventListener('hscredit-theme-change',draw);chart?.dispose()})
</script>
<template><div ref="root" class="chart" role="img" aria-label="交互式分析图表"/></template>
