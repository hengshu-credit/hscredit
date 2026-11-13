<script setup>
import {computed,nextTick,ref,watch,onMounted,onBeforeUnmount} from 'vue'
import {Minus, Plus, ScaleToOriginal} from '@element-plus/icons-vue'
import {apiBlob,artifactBlob} from '../api'
const props=defineProps({job:String,file:String,title:String,path:String,lazy:Boolean,tree:Boolean})
const isTree=computed(()=>props.tree||props.title==='树图'||props.title?.startsWith('报告预览 / tree_report.xlsx / '))
const zoom=ref(100), viewport=ref(null), aspect=ref(1)
const minZoom=25, maxZoom=400
const zoomProgress=computed(()=>100*(zoom.value-minZoom)/(maxZoom-minZoom)+'%')
async function setZoom(value){
  const el=viewport.value
  if(!el)return
  const center=(el.scrollLeft+el.clientWidth/2)/el.scrollWidth
  const top=el.scrollHeight?el.scrollTop/el.scrollHeight:0
  zoom.value=Math.max(minZoom,Math.min(maxZoom,Number(value)))
  await nextTick()
  el.scrollLeft=center*el.scrollWidth-el.clientWidth/2
  el.scrollTop=top*el.scrollHeight
}
const host=ref(null),visible=ref(!props.lazy)
let observer
onMounted(()=>{
  if (!props.lazy || !window.IntersectionObserver) {visible.value=true;return}
  observer=new IntersectionObserver(entries=>{
    if(entries.some(entry=>entry.isIntersecting)){visible.value=true;observer.disconnect()}
  },{rootMargin:'240px'})
  observer.observe(host.value)
})
const src=ref(''),error=ref('');let revision=0
watch(()=>[props.job,props.file,props.path,visible.value],async()=>{
  if(!visible.value)return
  const rev=++revision
  if(src.value)URL.revokeObjectURL(src.value)
  src.value='';error.value='';zoom.value=100;aspect.value=1
  try{
    const blob=await (props.path?apiBlob(props.path):artifactBlob(props.job,props.file))
    if(rev===revision)src.value=URL.createObjectURL(blob)
  }catch(e){if(rev===revision)error.value=e.message}
},{immediate:true})
onBeforeUnmount(()=>{revision++;observer?.disconnect();if(src.value)URL.revokeObjectURL(src.value)})
</script>
<template>
  <div ref="host" class="report-image" :class="{'tree-figure':isTree}">
    <div class="figure-heading"><h3>{{title}}</h3>
      <div class="figure-zoom" :class="{'zoom-pending':!src}" role="group" :aria-label="title+'缩放'">
        <button type="button" aria-label="缩小图片" title="缩小图片" :disabled="zoom===minZoom" @click="setZoom(zoom-25)"><Minus aria-hidden="true"/></button>
        <input class="zoom-slider" type="range" aria-label="图片缩放比例" :aria-valuetext="zoom+'%'" :min="minZoom" :max="maxZoom" step="5"
          :style="{'--progress':zoomProgress}" :value="zoom" @input="setZoom($event.target.value)"/>
        <button type="button" aria-label="放大图片" title="放大图片" :disabled="zoom===maxZoom" @click="setZoom(zoom+25)"><Plus aria-hidden="true"/></button>
        <span class="zoom-value">{{zoom}}%</span>
        <button type="button" class="zoom-fit" title="适应窗口" aria-label="适应窗口" :disabled="zoom===100" @click="setZoom(100)">
          <ScaleToOriginal aria-hidden="true"/><span>适应窗口</span>
        </button>
      </div>
    </div>
    <div v-if="src" ref="viewport" class="figure-viewport" tabindex="0" :aria-label="title" :style="{'--figure-scale':zoom/100,'--image-aspect':aspect}">
      <el-image :src="src" :preview-src-list="[src]" :alt="title" fit="contain" @load="aspect=$event.target.naturalWidth/$event.target.naturalHeight"/>
    </div>
    <el-alert v-else-if="error" :title="error" type="error"/><el-skeleton v-else :rows="5" animated/>
  </div>
</template>
<style scoped>
.report-image :deep(.el-image-viewer__btn){background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:var(--radius);opacity:1}
.report-image{--figure-height:min(58vh,560px);min-width:0}
.tree-figure{grid-column:1/-1}
.figure-heading{container-type:inline-size;display:flex;align-items:center;justify-content:space-between;gap:6px 12px;flex-wrap:wrap;margin-bottom:8px}
.figure-heading h3{margin:0;overflow-wrap:anywhere}
.figure-zoom{display:inline-flex;align-items:center;gap:2px;max-width:100%;padding:2px;margin-inline-start:auto;
  background:var(--surface);border:1px solid var(--border);border-radius:4px;color:var(--muted);font-size:12px;line-height:1}
.zoom-pending{visibility:hidden}
.figure-zoom button{display:inline-flex;align-items:center;justify-content:center;gap:5px;height:28px;min-width:28px;padding:0;border:0;border-radius:2px;color:inherit;background:transparent}
.figure-zoom button svg{width:14px;height:14px;fill:currentColor}
.figure-zoom button:enabled:hover,.figure-zoom button:enabled:focus-visible{color:var(--accent-text);background:var(--accent-soft)}
.figure-zoom button:disabled{color:color-mix(in srgb,var(--muted) 60%,var(--surface));cursor:default}
.zoom-slider{appearance:none;min-width:28px;width:76px;height:28px;margin:0 3px;padding:0;border:0;cursor:pointer;
  background:linear-gradient(to right,var(--accent-text) var(--progress),var(--border) var(--progress)) center / 100% 3px no-repeat}
.zoom-slider::-webkit-slider-thumb{appearance:none;width:12px;height:12px;border:2px solid var(--accent-text);border-radius:50%;background:var(--surface)}
.zoom-slider::-moz-range-thumb{box-sizing:border-box;width:12px;height:12px;border:2px solid var(--accent-text);border-radius:50%;background:var(--surface)}
.zoom-value{width:5ch;text-align:center;color:var(--text);font:600 12px/1 system-ui,sans-serif;font-variant-numeric:tabular-nums;flex:none}
.figure-zoom .zoom-fit{position:relative;margin-left:3px;padding:0 7px 0 9px;white-space:nowrap}
.zoom-fit::before{content:'';position:absolute;left:0;height:14px;border-left:1px solid var(--line-soft)}
@container (max-width:340px){.zoom-fit span{display:none}.figure-zoom .zoom-fit{padding-inline:6px}.zoom-slider{width:58px}}
@media (forced-colors:active){.zoom-slider{appearance:auto;background:revert}.zoom-slider::-webkit-slider-thumb{appearance:auto}}
.figure-viewport{overflow:auto;max-height:var(--figure-height)}
.figure-viewport .el-image{display:block;width:calc(min(100%,var(--figure-height) * var(--image-aspect)) * var(--figure-scale));margin-inline:auto}
.figure-viewport :deep(.el-image__inner){display:block;height:auto}
</style>
