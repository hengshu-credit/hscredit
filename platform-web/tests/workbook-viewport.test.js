import test from 'node:test'
import assert from 'node:assert/strict'
import {createSSRApp,h,ref} from 'vue'
import {renderToString} from '@vue/server-renderer'
import {COLS,columnWindows,pageAt,joinWindows,useWorkbookViewport} from '../src/workbook-viewport.js'

test('超出首个数据窗口的工作表会启用可滚动的完整列宽',async()=>{
  for(const width of [COLS,COLS+1,86]){
    const app=createSSRApp({async setup(){
      const viewport=useWorkbookViewport(ref(null),async()=>({width,height:1,first_row:1,first_col:1,rows:[[]],columns:['A'],cells:[],pictures:[]}))
      await viewport.open()
      assert.equal(viewport.wide.value,width>COLS)
      return ()=>h('div')
    }})
    await renderToString(app)
  }
})

test('滚动按可见列取数，直接拖到远端无需读中间列；页码按实际页高定位',()=>{
  assert.deepEqual(columnWindows(0,800,86),[1])
  assert.deepEqual(columnWindows(136*46,800,86),[37,49])
  assert.deepEqual([0,840,1220,99999].map(y=>pageAt([0,840,1220,3000],y)),[1,2,3,3])
})
test('相邻窗口衔接后只保留一个合并表头，跨行分页时保留本页跨度',()=>{
  const base={first_row:51,rows:Array(50).fill([]),columns:Array(12).fill(''),pictures:[]}
  const cell={row:51,text:'评估',merge:[11,45,14,52],rowspan:2,colspan:2}
  const result=joinWindows([{...base,first_col:1,cells:[{...cell,col:11}]},{...base,first_col:13,cells:[{...cell,col:13}]}])
  assert.equal(result.cells.length,1)
  assert.deepEqual([result.cells[0].col,result.cells[0].rowspan,result.cells[0].colspan],[11,2,4])
})
