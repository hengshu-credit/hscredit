// 手动跳页从目标页开始；用等高占位保留滚动位置，只挂载附近的字段行。
import {computed, ref, watch} from 'vue'

export const pageSizes = [10, 20, 50, 100]
let preferredSize = 20

export function useFieldPickerPages(fields) {
  const list = ref(null), size = ref(preferredSize), current = ref(1)
  const first = ref(1), last = ref(1)
  const totalPages = computed(() => Math.max(1, Math.ceil(fields.value.length / size.value)))
  const rowHeight=ref(36)
  const start=computed(()=>Math.max(first.value,current.value-1))
  const end=computed(()=>Math.min(last.value,current.value+1))
  const rows = computed(() => fields.value.slice((start.value - 1) * size.value, end.value * size.value))
  const before=computed(()=>(start.value-first.value)*size.value*rowHeight.value)
  const after=computed(()=>Math.max(0,Math.min(fields.value.length,last.value*size.value)-end.value*size.value)*rowHeight.value)

  function jump(page = 1) {
    const target = Math.max(1, Math.min(totalPages.value, page))
    first.value = last.value = current.value = target
    if (list.value) list.value.scrollTop = 0
  }
  function resize(value) {
    if (!pageSizes.includes(value)) return
    preferredSize = size.value = value
    jump()
  }
  function append() {
    if (last.value < totalPages.value) last.value++
  }
  function scroll() {
    const el = list.value
    const height = el?.querySelector('tbody tr[data-field-row]')?.getBoundingClientRect().height
    if (!height) return
    rowHeight.value=height
    const remaining = el.scrollHeight - el.clientHeight - el.scrollTop
    const topPage = first.value + Math.floor((el.scrollTop + 1) / (size.value * height))
    current.value = remaining <= 1 && el.scrollTop > 0 && last.value === totalPages.value
      ? last.value : Math.min(last.value, topPage)
    if (el.scrollTop > 0 && remaining < height) append()
  }
  function wheel(event) {
    const el = list.value
    if (event.deltaY > 0 && el && el.scrollHeight <= el.clientHeight + 1) append()
  }
  watch(totalPages, count => {
    if (current.value > count || first.value > count) jump(count)
    else last.value = Math.min(last.value, count)
  })
  return {list, size, current, rows, before, after, jump, resize, scroll, wheel}
}

