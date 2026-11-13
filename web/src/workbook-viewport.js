import {computed, nextTick, onBeforeUnmount, ref, shallowRef, shallowReactive, watch} from 'vue'
import {reportBlocks} from './workbook-layout.js'

export const ROWS = 50, COLS = 12, COLUMN_WIDTH = 136
export function columnWindows(left, width, count, span=COLS) {
  const start = Math.min(Math.ceil(count/span)-1, Math.floor(Math.max(0,left-96)/(span*COLUMN_WIDTH)))
  const last = Math.min(Math.ceil(count/span)-1, Math.floor((left+width+96)/(span*COLUMN_WIDTH)))
  return Array.from({length:Math.max(1,last-start+1)},(_,i)=>(start+i)*span+1)
}
export function pageAt(offsets, y) {
  let low=0, high=offsets.length-1
  while (low+1<high) {
    const mid=Math.floor((low+high)/2)
    if (offsets[mid]<=y) low=mid
    else high=mid
  }
  return low+1
}
export function joinWindows(parts) {
  const first=parts[0], last=parts.at(-1)
  const left=first.first_col, right=last.first_col+last.columns.length-1
  function joinCells(cells,top,bottom) {
    const result=[], merged=new Set()
    for (const cell of cells) {
      if (!cell.merge) {result.push(cell); continue}
      const [c,r,endCol,endRow]=cell.merge, key=r+':'+c
      if (merged.has(key)) continue
      merged.add(key)
      const col=Math.max(c,left), row=Math.max(r,top)
      result.push({...cell,col,row,colspan:Math.min(endCol,right)-col+1,rowspan:Math.min(endRow,bottom)-row+1})
    }
    return result
  }
  const top=first.first_row, bottom=last.first_row+last.rows.length-1
  const cells=joinCells(parts.flatMap(p=>p.cells),top,bottom)
  const header=first.header_context
  const header_context=header && {...header,cells:joinCells(parts.filter(p=>p.first_row===top).flatMap(p=>p.header_context?.cells||[]),header.first_row,header.last_row)}
  const vertical=first.first_row!==last.first_row
  return {...first, rows:vertical?parts.flatMap(p=>p.rows):first.rows,
    columns:vertical?first.columns:parts.flatMap(p=>p.columns), cells,header_context,
    data_rows:vertical?parts.flatMap(p=>p.data_rows||[]):first.data_rows,pictures:parts.flatMap(p=>p.pictures)}
}

export function useWorkbookViewport(pane, fetchWindow) {
  const meta=shallowRef(null), error=ref(''), pending=ref(0), x=ref(0), y=ref(0), vw=ref(800), vh=ref(480)
  const bands=shallowReactive(new Map()), heights=shallowReactive(new Map())
  const cache=new Map(), requests=new Map()
  let controller=new AbortController(), desired='', navigation=0, frame=0, observer
  const count=computed(()=>Math.ceil((meta.value?.height||1)/ROWS))
  // 超过首个数据窗口就需要横向滚动轨道，否则后续列无法进入视口。
  const wide=computed(()=>(meta.value?.width||0)>(meta.value?.windowSpan||COLS))
  const offsets=computed(()=>{
    const a=[0]
    for(let p=1;p<=count.value;p++) a.push(a[p-1]+(heights.get(p)??1000))
    return a
  })
  const currentPage=computed(()=>pageAt(offsets.value,y.value+1))
  // 连续滚过交界后仍保留相邻页的真实表头、列宽和滚动容器。
  const firstVisible=computed(()=>Math.max(1,pageAt(offsets.value,Math.max(0,y.value-180))-1))
  const lastVisible=computed(()=>Math.min(count.value,pageAt(offsets.value,y.value+vh.value+180)+1))
  const visible=computed(()=>{
    const start=firstVisible.value, end=lastVisible.value
    return Array.from({length:end-start+1},(_,i)=>start+i)
  })
  // 将相邻行窗口交给同一次工作表排版，读取批次不产生表框、滚动条或重复表头。
  const papers=computed(()=>{
    const runs=[]
    for(const page of visible.value){
      const band=bands.get(page)
      if(!band)continue
      const previous=runs.at(-1)
      if(previous && previous.last===page-1 && previous.key===band.key){
        previous.last=page
        previous.parts.push(band.data)
      }else runs.push({first:page,last:page,key:band.key,parts:[band.data]})
    }
    return runs.map(({first,last,parts})=>({first,last,blocks:reportBlocks(parts.length===1?parts[0]:joinWindows(parts))}))
  })
  function clear(){
    controller.abort();controller=new AbortController(); navigation++;desired=''
    cache.clear();requests.clear();bands.clear();heights.clear()
    meta.value=null;error.value='';pending.value=0;x.value=0;y.value=0
    if(pane.value)pane.value.scrollTo(0,0)
  }
  async function tile(page,col){
    const key=page+':'+col, signal=controller.signal
    if(cache.has(key)){
      const hit=cache.get(key);cache.delete(key);cache.set(key,hit);return hit
    }
    if(requests.has(key))return requests.get(key)
    pending.value++
    const work=fetchWindow(page,col,signal).then(data=>{
      if(signal.aborted)return null
      cache.set(key,data)
      if(cache.size>80)cache.delete(cache.keys().next().value)
      return data
    }).finally(()=>{if(!signal.aborted){requests.delete(key);pending.value--}})
    requests.set(key,work);return work
  }
  async function open(){
    clear()
    const signal=controller.signal
    try{
      const data=await tile(1,1)
      if(!data||signal.aborted)return
      meta.value=data
      bands.set(1,{key:'1',data})
      await nextTick();sync()
    }catch(e){if(!signal.aborted)error.value=e.message}
  }
  function sync(){
    const el=pane.value
    if(!el)return
    x.value=el.scrollLeft;y.value=el.scrollTop;vw.value=el.clientWidth;vh.value=el.clientHeight
  }
  function scroll(){
    if(!frame)frame=requestAnimationFrame(()=>{frame=0;sync()})
  }
  function measure(paper,el){
    if(!meta.value||!el.offsetHeight||!el.isConnected)return
    const top=el.getBoundingClientRect().top
    const starts=[0]
    const elements=[...el.querySelectorAll('[data-row]')]
    for(let page=paper.first+1;page<=paper.last;page++){
      const firstRow=(page-1)*ROWS+1
      const next=elements.find(node=>Number(node.dataset.row)>=firstRow)
      const boundary=next?Math.round(next.getBoundingClientRect().top-top):el.offsetHeight
      starts.push(Math.max(starts.at(-1),Math.min(boundary,el.offsetHeight)))
    }
    starts.push(el.offsetHeight)
    const current=currentPage.value
    const oldTop=offsets.value[current-1]
    for(let i=0;i<starts.length-1;i++)heights.set(paper.first+i,Math.max(1,starts[i+1]-starts[i]))
    const delta=offsets.value[current-1]-oldTop
    if(delta && pane.value){pane.value.scrollTop+=delta;sync()}
  }
  const windowsAt=left=>columnWindows(left,vw.value,meta.value?.width||1,meta.value?.windowSpan||COLS)
  const windows=computed(()=>windowsAt(x.value))
  async function loadBand(p,cols,valid){
    const key=cols.join(',')
    if(bands.get(p)?.key===key)return
    const parts=await Promise.all(cols.map(c=>tile(p,c)))
    if(valid()&&parts.every(Boolean))bands.set(p,{key,data:joinWindows(parts)})
  }
  async function refresh(){
    if(!meta.value)return
    const pages=visible.value, cols=windows.value, signal=controller.signal
    const key=pages.join(',')+'/'+cols.join(',')
    desired=key
    const valid=()=>!signal.aborted&&desired===key
    try{
      await Promise.all(pages.map(p=>loadBand(p,cols,valid)))
      if(valid())for(const p of bands.keys())if(!pages.includes(p))bands.delete(p)
    }catch(e){if(valid())error.value=e.message}
  }
  watch(()=>meta.value?visible.value.join(',')+'/'+windows.value.join(','):'',(key,_,cleanup)=>{
    desired=key;error.value=''
    if(!key)return
    // 拖动滚动条经过多个窗口时，只给停留的窗口发请求。
    const timer=setTimeout(refresh,40)
    cleanup(()=>clearTimeout(timer))
  },{flush:'post'})
  async function goTo(page,col=null,row=null){
    if(!meta.value)return
    const token=++navigation, signal=controller.signal
    const left=col===null?x.value:Math.min(Math.max(0,(col-1)*COLUMN_WIDTH),Math.max(0,meta.value.width*COLUMN_WIDTH-vw.value))
    const cols=windowsAt(left)
    try{
      error.value=''
      await loadBand(page,cols,()=>token===navigation&&!signal.aborted)
      if(token!==navigation||signal.aborted||!pane.value)return
      pane.value.scrollTo(wide.value?left:0,offsets.value[page-1]);sync()
      await nextTick()
      if(row && token===navigation){
        const band=pane.value
        const rowSelector='[data-row="'+row+'"]'
        const cell=col && [...(band?.querySelectorAll(rowSelector+' > [data-col]') || [])].find(el=>{
          const start=Number(el.dataset.col)
          return col>=start && col<start+el.colSpan
        })
        const target=cell || band?.querySelector(rowSelector)
        if(cell && !wide.value){
          const table=cell.closest('.report-table-scroll')
          table.scrollLeft+=cell.getBoundingClientRect().left-table.getBoundingClientRect().left
        }
        if(target){pane.value.scrollTop+=target.getBoundingClientRect().top-pane.value.getBoundingClientRect().top;sync()}
      }
    }catch(e){if(token===navigation&&!signal.aborted)error.value=e.message}
  }
  watch(pane,el=>{
    observer?.disconnect()
    if(el){observer=new ResizeObserver(scroll);observer.observe(el);sync()}
  },{flush:'post'})
  onBeforeUnmount(()=>{controller.abort();observer?.disconnect();cancelAnimationFrame(frame)})
  return {meta,papers,offsets,wide,vw,currentPage,pending,error,open,clear,refresh,goTo,measure,scroll}
}
