<script setup>
import {computed, nextTick, onBeforeUnmount, onMounted, ref, shallowReactive, watch} from 'vue'

const props=defineProps({names:{type:Array,required:true},placeholder:String})
const viewport=ref(null), width=ref(300), top=ref(0)
const heights=shallowReactive(new Map())
const batch=16, lineHeight=20, maxHeight=80
let observer
// 每块用一个可换行文本节点承载多个字段。浏览器只排版可见块及相邻缓冲块。
const chunks=computed(()=>{
  const result=[]
  for(let i=0;i<props.names.length;i+=batch)result.push(props.names.slice(i,i+batch).join('、'))
  return result
})
const offsets=computed(()=>{
  const result=[0]
  for(let i=0;i<chunks.value.length;i++){
    const estimate=lineHeight*Math.max(1,Math.ceil(chunks.value[i].length*8/Math.max(1,width.value)))
    result.push(result[i]+(heights.get(i)??estimate))
  }
  return result
})
function at(y){
  let lo=0, hi=chunks.value.length
  while(lo+1<hi){const mid=Math.floor((lo+hi)/2);if(offsets.value[mid]<=y)lo=mid;else hi=mid}
  return lo
}
const start=computed(()=>Math.max(0,at(top.value)-1))
const end=computed(()=>Math.min(chunks.value.length,at(top.value+maxHeight)+2))
const visible=computed(()=>chunks.value.slice(start.value,end.value))
function scroll(){top.value=viewport.value?.scrollTop||0}
function page(direction){viewport.value?.scrollBy(0,direction*maxHeight)}
defineExpose({page})
const vMeasure={
  mounted(el){observer?.observe(el)},
  unmounted(el){observer?.unobserve(el)},
}
onMounted(()=>{
  observer=new ResizeObserver(entries=>{
    if(!viewport.value)return
    const w=viewport.value.clientWidth
    if(w>0 && w!==width.value){width.value=w;heights.clear()}
    let adjustment=0
    for(const {target} of entries){
      if(target===viewport.value||!target.isConnected)continue
      const index=Number(target.dataset.chunk), height=target.offsetHeight
      if(!height||index>=chunks.value.length)continue
      if(offsets.value[index+1]<=top.value)adjustment+=height-(offsets.value[index+1]-offsets.value[index])
      heights.set(index,height)
    }
    if(adjustment)nextTick(()=>{if(viewport.value)viewport.value.scrollTop+=adjustment})
  })
  observer.observe(viewport.value)
  viewport.value.querySelectorAll('[data-chunk]').forEach(el=>observer.observe(el))
})
watch(chunks,()=>{heights.clear();top.value=0;if(viewport.value)viewport.value.scrollTop=0},{flush:'sync'})
onBeforeUnmount(()=>observer?.disconnect())
</script>
<template>
  <span ref="viewport" class="selected-names" @scroll.passive="scroll">
    <span v-if="!names.length" class="muted">{{placeholder}}</span>
    <span v-else class="names-window" :style="{paddingTop:offsets[start]+'px',paddingBottom:(offsets.at(-1)-offsets[end])+'px'}">
      <span v-for="(chunk,i) in visible" :key="start+i" v-measure :data-chunk="start+i">{{chunk}}</span>
    </span>
  </span>
</template>
<style scoped>
.selected-names{display:block;flex:1;min-width:0;max-height:80px;overflow:auto;overscroll-behavior:contain;overflow-anchor:none;text-align:left;white-space:normal;overflow-wrap:anywhere;line-height:20px;font-size:13px}
.names-window,.names-window>span{display:block}.selected-names>.muted{line-height:inherit}
</style>
