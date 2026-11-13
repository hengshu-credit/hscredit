import test from 'node:test'
import assert from 'node:assert/strict'
import {createSSRApp,h,ref} from 'vue'
import {renderToString} from '@vue/server-renderer'
import {COLS,columnWindows,pageAt,joinWindows,useWorkbookViewport} from '../src/workbook-viewport.js'
import {reportBlocks} from '../src/workbook-layout.js'

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

test('跨行窗口的矩阵只显示一次表头，稀疏行和跨页合并保留，真实空行分表',()=>{
  const cell=(row,col,text,options={})=>({row,col,text,colspan:1,rowspan:1,...options})
  const header=[cell(49,65,'甲',{bold:true}),cell(49,66,'乙',{bold:true})]
  const merged={merge:[65,50,65,51],rowspan:1}
  const base={first_col:65,columns:['BM','BN'],pictures:[]}
  const parts=[
    {...base,first_row:1,rows:Array(50).fill([]),cells:[...header,cell(50,65,'样本',merged),cell(50,66,'1')]},
    {...base,first_row:51,rows:Array(6).fill([]),header_context:{first_row:49,last_row:49,cells:header},
      data_rows:[51,52,53,55,56],cells:[cell(51,65,'样本',merged),cell(51,66,'2'),cell(53,66,'3'),
        cell(55,65,'另一张表',{bold:true}),cell(55,66,'值',{bold:true}),cell(56,66,'4')]},
  ]
  const [matrix,other]=reportBlocks(joinWindows(parts))
  assert.deepEqual(matrix.rows.map(r=>r.row),[49,50,51,52,53])
  assert.deepEqual(matrix.rows.filter(r=>r.header).map(r=>r.row),[49])
  assert.equal(matrix.rows[1].cells[0].rowspan,2)
  assert.deepEqual(matrix.rows[2].cells.map(c=>c.text),['2'])
  assert.deepEqual(matrix.rows[3].cells.map(c=>c.text),['',''])
  assert.deepEqual(other.rows.map(r=>r.row),[55,56])
})
