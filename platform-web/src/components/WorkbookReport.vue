<script setup>
import {computed, nextTick, onBeforeUnmount, onMounted, ref, watch} from 'vue'
import {ElMessage} from 'element-plus'
import {Document, Download, FullScreen, Grid, List} from '@element-plus/icons-vue'
import {api, download} from '../api'
import {ROWS, COLUMN_WIDTH, useWorkbookViewport} from '../workbook-viewport'
import ReportImage from './ReportImage.vue'

const props = defineProps({job:{type:Object,required:true}, foldable:Boolean, preferredFile:String})
const root = ref(null), opened = ref(!props.foldable), fullscreen = ref(false)
const files = computed(() => (props.job.artifacts || []).filter(name=>name.endsWith('.xlsx')))
const file=ref(''), sheet=ref(''), pane=ref(null), saving=ref(false)
const path=computed(()=>'/jobs/'+props.job.id+'/workbooks/'+encodeURIComponent(file.value))
// 一个窗口容纳常规订单与金额宽表，更宽的工作表仍按可见列加载。
const windowSpan=64
const {meta:preview,bands,offsets,visible,wide,vw,currentPage,pending,error,open,clear,refresh,goTo,measure,scroll}=useWorkbookViewport(pane,async(page,col,signal)=>{
  const query=new URLSearchParams({page,first_col:col,column_count:windowSpan,column_page:Math.ceil(col/30),include_catalog:page===1&&col===1})
  if(sheet.value)query.set('sheet',sheet.value)
  const data=await api(path.value+'?'+query,{signal})
  if(data.first_col)return {...data,windowSpan}
  // 仍在运行的旧服务以每 30 列返回窗口，页面继续使用连续滚动。
  const cells=data.cells??data.rows.flatMap((row,i)=>row.flatMap((text,j)=>text===''?[]:[{text,row:data.first_row+i,col:col+j,colspan:1,rowspan:1}]))
  const pictures=data.pictures??(page===1&&col===1?Array.from({length:data.image_count},(_,index)=>({row:1,col:1,index})):[])
  return {...data,first_col:col,windowSpan:30,cells,pictures}
})
const sheets=computed(()=>preview.value?.sheets||[])
const activeSheet=computed(()=>preview.value?.sheet)
function selectFile(value){file.value=value;sheet.value=''}
watch([()=>props.job.id,()=>files.value.join('\n'),()=>props.preferredFile],()=>{
  selectFile(files.value.includes(props.preferredFile)?props.preferredFile:files.value[0])
},{immediate:true})
let destination=null
watch([path,sheet,opened],async(_,__,onCleanup)=>{
  let stale=false;onCleanup(()=>{stale=true})
  if(!opened.value||!file.value){clear();return}
  await open()
  if(!stale&&preview.value&&destination?.sheet===activeSheet.value){
    const link=destination;destination=null;await jump(link)
  }
},{immediate:true})
function selectSheet(value){if(value!==activeSheet.value){destination=null;sheet.value=value}}
function jump(link){
  const col=[...link.col].reduce((n,c)=>n*26+c.charCodeAt(0)-64,0)
  return goTo(Math.ceil(link.row/ROWS),col,link.row)
}
function follow(link){
  if(activeSheet.value===link.sheet)return jump(link)
  destination=link;sheet.value=link.sheet
}
const vMeasure={
  mounted(el,{value}){
    const observer=new ResizeObserver(()=>measure(value,el.offsetHeight))
    el._sizeObserver=observer;observer.observe(el)
  },
  unmounted(el){el._sizeObserver?.disconnect()},
}
function blockStyle(block){
  if(!wide.value)return {}
  return {width:vw.value+'px',position:'sticky',left:'0px'}
}
function tableStyle(table,block){
  if (!wide.value) return {}
  const previous=block.tables?.[block.tables.indexOf(table)-1]
  const start=previous?previous.left+previous.widths.length:1
  return {marginLeft:(table.left-start)*COLUMN_WIDTH+'px',width:table.widths.length*COLUMN_WIDTH+'px'}
}
function retry(){error.value='';if(preview.value)refresh();else open()}
async function save() {
  saving.value = true
  try { await download(props.job.id,file.value) } catch (e) { ElMessage.error(e.message) }
  finally { saving.value = false }
}
const canExpand = !!document.fullscreenEnabled
async function expand() {
  try {
    if (document.fullscreenElement===root.value) await document.exitFullscreen()
    else await root.value.requestFullscreen()
  } catch { ElMessage.error('浏览器未能打开全屏阅读') }
}
function onFullscreen(){ fullscreen.value = document.fullscreenElement===root.value }
onMounted(()=>document.addEventListener('fullscreenchange',onFullscreen))
onBeforeUnmount(()=>document.removeEventListener('fullscreenchange',onFullscreen))
</script>
<template>
  <section v-if="files.length&&job.status==='成功'" ref="root" class="workbook" aria-label="Excel 报告">
    <header class="reader-toolbar">
      <div class="file-title"><span class="file-icon"><el-icon><Document/></el-icon></span><div>
        <h2>Excel 报告</h2><span v-if="files.length===1" class="filename">{{file}}</span>
        <el-select v-else :model-value="file" aria-label="报告文件" size="small" filterable :teleported="false" @change="selectFile">
          <el-option v-for="name in files" :key="name" :value="name"/>
        </el-select>
      </div></div>
      <div class="reader-actions">
        <el-button v-if="foldable&&!fullscreen" :aria-expanded="opened" @click="opened=!opened">{{opened?'收起报告':'查看报告'}}</el-button>
        <el-button v-if="canExpand&&opened" :icon="FullScreen" :aria-label="fullscreen?'退出全屏':'全屏阅读'" :title="fullscreen?'退出全屏':'全屏阅读'" @click="expand"/>
        <el-button type="primary" :icon="Download" :loading="saving" @click="save">下载 Excel 报告</el-button>
      </div>
    </header>
    <div v-if="opened" class="reader-layout">
      <nav v-if="sheets.length>1" class="sheet-nav" aria-label="工作表目录">
        <div class="nav-heading"><el-icon><List/></el-icon>工作表<span>{{sheets.length}}</span></div>
        <div class="sheet-list"><button v-for="name in sheets" :key="name" type="button" :aria-current="activeSheet===name?'page':undefined" :title="name" @click="selectSheet(name)">
          <el-icon><Grid/></el-icon><span>{{name}}</span>
        </button></div>
        <el-select class="mobile-sheets" :model-value="activeSheet" aria-label="选择工作表" filterable :teleported="false" @change="selectSheet"><el-option v-for="name in sheets" :key="name" :value="name"/></el-select>
      </nav>
      <div class="reader-main" :aria-busy="pending>0">
        <div v-if="error" class="reader-error"><el-alert :title="error" type="error" :closable="false"/><el-button @click="retry">重新加载</el-button></div>
        <div v-if="!preview&&!error" class="reader-feedback" role="status">正在读取工作表…</div>
        <template v-if="preview">
          <div class="sheet-title"><h3>{{preview.sheet}}</h3>
            <el-select v-if="preview.outline?.length>1" class="chapter-picker" aria-label="跳至章节" placeholder="跳至章节" filterable
              :model-value="null" :teleported="false" @change="i=>follow({...preview.outline[i],sheet:preview.sheet})">
              <el-option v-for="(item,i) in preview.outline" :key="item.row" :value="i" :label="item.text"/>
            </el-select>
            <span>{{preview.height.toLocaleString()}} 行 · {{preview.width}} 列</span>
          </div>
          <article ref="pane" class="sheet-viewport" :style="{'--content-height':offsets.at(-1)+'px'}" tabindex="0" :aria-label="preview.sheet+'工作表内容'" @scroll.passive="scroll">
            <div class="sheet-canvas" :style="{height:offsets.at(-1)+'px',width:wide?preview.width*COLUMN_WIDTH+'px':'100%'}">
            <section v-for="p in visible" :key="preview.sheet+':'+p" class="sheet-band" :style="{top:offsets[p-1]+'px'}" :data-page="p">
              <div v-if="bands.has(p)" v-measure="p" class="report-paper" :class="{'wide-paper':wide}">
              <p v-if="!bands.get(p).blocks.length" class="muted empty-window">当前区域为空</p>
              <template v-for="block in bands.get(p).blocks" :key="block.type+block.row">
              <h4 v-if="block.type==='heading'" class="report-heading" :style="blockStyle(block)" :data-row="block.row">
                <button v-if="block.cell.link" type="button" class="sheet-link" @click="follow(block.cell.link)">{{block.cell.text}}</button>
                <template v-else>{{block.cell.text}}</template>
              </h4>
              <p v-else-if="block.type==='paragraph'" class="report-paragraph" :style="blockStyle(block)" :data-row="block.row">
                <button v-if="block.cell.link" type="button" class="sheet-link" @click="follow(block.cell.link)">{{block.cell.text}}</button>
                <template v-else>{{block.cell.text}}</template>
              </p>
              <div v-else-if="block.type==='pictures'" class="report-figures" :style="blockStyle(block)" :data-row="block.row">
                <ReportImage v-for="p in block.pictures" :key="p.index+'-'+p.col+'-'+p.row" lazy :path="path+'/images/'+p.index+'?sheet='+encodeURIComponent(preview.sheet)" :title="preview.sheet+' · 图 '+(p.index+1)"/>
              </div>
              <div v-else class="report-tables" :class="{parallel:block.tables}">
              <div v-for="part in (block.tables || [block])" :key="part.left" class="report-table-scroll" :style="tableStyle(part,block)" tabindex="0" role="region" :aria-label="preview.sheet+'表格'">
                <table :style="{minWidth:wide?'100%':part.minWidth+'px'}"><colgroup><col v-for="(w,i) in part.widths" :key="i" :style="{width:(wide?100/part.widths.length:100*w/part.minWidth)+'%'}"/></colgroup><tbody><tr v-for="r in part.rows" :key="r.row" :data-row="r.row" :class="{'report-table-header':r.header}">
                  <component :is="r.header?'th':'td'" v-for="c in r.cells" :key="c.col" :scope="r.header?(c.colspan>1?'colgroup':'col'):undefined"
                    :rowspan="c.rowspan" :colspan="c.colspan" :data-col="c.col" :style="{textAlign:c.align}" :class="{'number-cell':c.numeric,emphasis:c.bold}" :title="c.text">
                    <button v-if="c.link" type="button" class="sheet-link" @click="follow(c.link)">{{c.text}}</button>
                    <template v-else>{{c.display??c.text}}</template>
                  </component>
                </tr></tbody></table>
              </div>
              </div>
              </template>
              </div>
            </section>
            </div>
          </article>
          <footer class="reader-paging">
            <span>行 {{(currentPage-1)*ROWS+1}}–{{Math.min(preview.height,currentPage*ROWS)}} / {{preview.height}}</span>
            <span role="status" class="load-state">{{pending?'正在读取…':''}}</span>
            <el-pagination v-if="preview.height>ROWS" :current-page="currentPage" :page-size="ROWS" :total="preview.height" :pager-count="5" size="small"
              layout="prev,pager,next" aria-label="工作表行分页" @update:current-page="goTo($event)"/>
          </footer>
        </template>
      </div>
    </div>
  </section>
</template>
<style scoped src="./workbook-report.css"></style>
