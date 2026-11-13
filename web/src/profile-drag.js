// 数据预览和字段选择器共用标题栏拖拽，浮层位置使用视口坐标。
import {computed, onBeforeUnmount, ref} from 'vue'

export function useProfileDrag(pinned, panelSelector) {
  const position = ref(null), dragging = ref(false)
  const floatingStyle = computed(() => position.value ? {
    '--profile-left':position.value.left+'px', '--profile-top':position.value.top+'px'
  } : {})
  let panel, gesture, observer

  function moveTo(left,top) {
    if (!panel) return
    const {width,height} = panel.getBoundingClientRect()
    const viewport = document.documentElement
    position.value = {
      left:Math.max(8, Math.min(left, viewport.clientWidth-width-8)),
      top:Math.max(8, Math.min(top, viewport.clientHeight-height-8))
    }
  }
  function keepInViewport() {
    if (position.value) moveTo(position.value.left,position.value.top)
  }
  function startDrag(event) {
    if (event.button !== 0 || event.isPrimary === false || event.target.closest('.profile-actions')) return
    panel = event.currentTarget.closest(panelSelector)
    if (!panel) return
    const {left,top} = panel.getBoundingClientRect()
    gesture = {handle:event.currentTarget,id:event.pointerId,x:event.clientX,y:event.clientY,left,top,moved:false}
    dragging.value = true
    gesture.handle.setPointerCapture(gesture.id)
    event.preventDefault()
  }
  function moveDrag(event) {
    if (!gesture || event.pointerId !== gesture.id) return
    const dx = event.clientX-gesture.x, dy = event.clientY-gesture.y
    if (!gesture.moved) {
      if (Math.hypot(dx,dy) < 3) return
      gesture.moved = true
      pinned.value = true
      if (!observer) {
        observer = new ResizeObserver(keepInViewport)
        observer.observe(panel)
        window.addEventListener('resize',keepInViewport)
      }
    }
    moveTo(gesture.left+dx,gesture.top+dy)
  }
  function stopDrag() {
    if (!gesture) return
    const {handle,id} = gesture
    gesture = null
    dragging.value = false
    if (handle.hasPointerCapture(id)) handle.releasePointerCapture(id)
  }
  function resetPosition() {
    stopDrag()
    observer?.disconnect()
    observer = null
    window.removeEventListener('resize',keepInViewport)
    position.value = null
    panel = null
  }
  onBeforeUnmount(resetPosition)
  return {position, dragging, floatingStyle, startDrag, moveDrag, stopDrag, resetPosition, keepInViewport}
}

