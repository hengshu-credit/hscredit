<script setup>
import {ref,watch,onBeforeUnmount} from 'vue'
import {ArrowRight,DArrowLeft,DArrowRight} from '@element-plus/icons-vue'
import BrandLockup from './BrandLockup.vue'
import {SIDEBAR_MIN_WIDTH,SIDEBAR_MAX_WIDTH,SIDEBAR_DEFAULT_WIDTH,clampSidebarWidth} from '../layout'
const props=defineProps({menus:Array,active:String,width:Number,collapsed:Boolean,resizable:Boolean,top:Boolean})
const emit=defineEmits(['navigate','toggle','resize','resize-end'])
const resizing=ref(false)
let drag=null
function startResize(event){
  if(!props.resizable||event.button!==0)return
  event.preventDefault()
  event.currentTarget.focus({preventScroll:true})
  drag={element:event.currentTarget,id:event.pointerId,x:event.clientX,width:props.width,next:props.width}
  drag.element.setPointerCapture(drag.id)
  resizing.value=true
  document.documentElement.classList.add('sidebar-resizing')
  window.addEventListener('blur',cancelResize)
}
function moveResize(event){
  if(!drag||event.pointerId!==drag.id)return
  drag.next=clampSidebarWidth(drag.width+event.clientX-drag.x)
  emit('resize',drag.next)
}
function finishResize(cancel=false){
  if(!drag)return
  const previous=drag
  drag=null;resizing.value=false
  document.documentElement.classList.remove('sidebar-resizing')
  window.removeEventListener('blur',cancelResize)
  if(previous.element.hasPointerCapture(previous.id))previous.element.releasePointerCapture(previous.id)
  emit('resize-end',cancel?previous.width:previous.next)
}
function cancelResize(){finishResize(true)}
function resizeKey(event){
  if(event.key==='Escape'&&drag){event.preventDefault();finishResize(true);return}
  const value={ArrowLeft:props.width-16,ArrowRight:props.width+16,Home:SIDEBAR_MIN_WIDTH,End:SIDEBAR_MAX_WIDTH}[event.key]
  if(value===undefined)return
  event.preventDefault();emit('resize-end',clampSidebarWidth(value))
}
watch(()=>props.resizable,value=>{if(!value)finishResize(true)})
onBeforeUnmount(()=>finishResize(true))
</script>
<template>
  <aside id="platform-sidebar" class="sidebar" :class="{'is-resizing':resizing}" aria-label="工作台导航">
    <BrandLockup/>
    <nav aria-label="主菜单">
      <button v-for="menu in menus" :key="menu.id" type="button" :class="{active:active===menu.id}" :title="menu.name" :aria-label="menu.name" :aria-current="active===menu.id?'page':undefined" @click="emit('navigate',menu.id)">
        <el-icon><component :is="menu.icon"/></el-icon><span>{{menu.name}}</span><el-icon v-if="active===menu.id" class="nav-arrow"><ArrowRight/></el-icon>
      </button>
    </nav>
    <div v-if="!top" class="sidebar-footer"><span class="status-dot"/>本地工作空间</div>
    <button v-if="!top" type="button" class="sidebar-collapse" :aria-label="collapsed?'展开侧边栏':'收起侧边栏'" :title="collapsed?'展开侧边栏':'收起侧边栏'" :aria-expanded="!collapsed" aria-controls="platform-sidebar" @click="emit('toggle')">
      <el-icon><DArrowRight v-if="collapsed"/><DArrowLeft v-else/></el-icon>
    </button>
    <div v-if="resizable" role="separator" aria-label="调整侧边栏宽度" aria-orientation="vertical" :aria-valuemin="SIDEBAR_MIN_WIDTH" :aria-valuemax="SIDEBAR_MAX_WIDTH" :aria-valuenow="width" :aria-valuetext="`${width} 像素`" tabindex="0" class="sidebar-resizer" title="拖拽调整宽度；双击恢复默认；方向键微调" @pointerdown="startResize" @pointermove="moveResize" @pointerup="finishResize()" @pointercancel="finishResize(true)" @lostpointercapture="finishResize(true)" @keydown="resizeKey" @dblclick="emit('resize-end',SIDEBAR_DEFAULT_WIDTH)"/>
  </aside>
</template>
