import test from 'node:test'
import assert from 'node:assert/strict'
import {reportBlocks} from '../src/workbook-layout.js'

const cell = (row,col,text,options={}) => ({row,col,text,bold:false,numeric:false,rowspan:1,colspan:1,...options})
test('报告目录合并标题、跨行列的表头、图版按原顺序呈现', () => {
  const page = {first_row:1, rows:Array(12).fill([]), pictures:[{row:11,col:2,index:0}],
    cells:[cell(2,2,'评估报告',{bold:true,colspan:4}),cell(4,2,'指标',{bold:true,rowspan:2}),
      cell(4,3,'MOB6',{bold:true,colspan:2}),cell(5,3,'训练',{bold:true}),cell(5,4,'测试',{bold:true}),
      cell(6,2,'AUC'),cell(6,3,'0.81',{numeric:true}),cell(6,4,'0.77',{numeric:true})]}
  const blocks = reportBlocks(page)
  assert.deepEqual(blocks.map(b=>b.type),['heading','table','pictures'])
  assert.deepEqual(blocks[1].rows.map(r=>r.cells.map(c=>c.text)),[['指标','MOB6'],['训练','测试'],['AUC','0.81','0.77']])
  assert.deepEqual(blocks[1].rows.map(r=>r.header),[true,true,false])
  assert.equal(blocks[1].rows[0].cells[0].rowspan,2)
  assert.equal(blocks[2].pictures[0].row,11)
})

test('页内有效数据靠原列对齐，空白间隔单独分段', () => {
  const page = {first_row:51, rows:Array(8).fill([]), pictures:[],
    cells:[cell(51,32,'样本量'),cell(52,33,'42',{numeric:true}),cell(57,32,'说明')]}
  const blocks = reportBlocks(page)
  assert.deepEqual(blocks[0].rows.map(r=>r.cells.map(c=>c.text)),[['样本量',''],['','42']])
  assert.equal(blocks[1].type,'paragraph')
})

test('并列分箱独立保留口径标题，两级表头按父指标拆开旧版跨组合并', () => {
  const head=(r,c,text,colspan=1)=>cell(r,c,text,{bold:true,colspan})
  const page={first_row:1, rows:Array(6).fill([]), cells:[
    head(1,2,'2、模型尾部区分能力',7),head(2,2,'订单口径',3),head(2,6,'金额口径',3),
    head(4,2,'数据集'),head(4,3,'KS'),head(4,4,'AUC'),head(4,6,'数据集'),head(4,7,'KS'),head(4,8,'AUC'),
    head(5,2,'样本'),head(5,3,'MOB6',2),head(5,6,'样本'),head(5,7,'MOB6',2),
    cell(6,2,'训练'),cell(6,3,'0.3'),cell(6,4,'0.7'),cell(6,6,'训练'),cell(6,7,'0.2'),cell(6,8,'0.6'),
  ]}
  const [section, parallel] = reportBlocks(page)
  assert.equal(section.type,'heading')
  assert.equal(parallel.type,'parallel')
  assert.deepEqual(parallel.tables.map(t=>t.left),[2,6])
  for (const [i,t] of parallel.tables.entries()) {
    assert.equal(t.rows[0].cells[0].text,i===0?'订单口径':'金额口径')
    assert.deepEqual(t.rows[2].cells.map(c=>[c.text,c.colspan]),[['样本',1],['MOB6',1],['MOB6',1]])
    assert.equal(t.rows[3].header,false)
  }
})

test('跨页分箱保留前页的口径和多层表头', () => {
  const head=(r,c,text,colspan=1)=>cell(r,c,text,{bold:true,colspan})
  const page={first_row:51,rows:[[]],cells:[cell(51,2,'10'),cell(51,3,'0.12')],header_context:{
    first_row:45,last_row:46,cells:[head(45,2,'订单口径',2),head(46,2,'样本数'),head(46,3,'坏样本率')],
  }}
  const [table]=reportBlocks(page)
  assert.deepEqual(table.rows.map(r=>r.row),[45,46,51])
  assert.deepEqual(table.rows.map(r=>r.header),[true,true,false])
})

test('三级表头仅为有下一级的标题添加分组标记，跨行叶标题独立显示', () => {
  const header=(row,col,text,colspan=1,rowspan=1)=>cell(row,col,text,{bold:true,colspan,rowspan})
  const page={first_row:51,rows:[[]],header_context:{first_row:44,last_row:46,cells:[
    header(44,2,'数据集',1,3),header(44,3,'样本数',4),header(44,7,'样本率',2),
    header(45,3,'好样本',2),header(45,5,'坏样本',2),header(45,7,'坏样本率',2),
    ...Array.from({length:6},(_,i)=>header(46,3+i,i%2?'MOB1_30+':'MOB6_30+')),
  ]},cells:[cell(51,2,'训练集'),...Array.from({length:6},(_,i)=>cell(51,3+i,'10',{numeric:true}))]}
  const [table]=reportBlocks(page)
  assert.deepEqual(table.rows.map(row=>row.groupedHeader),[true,true,true,false])
  assert.deepEqual(table.rows.map(row=>row.cells.filter(c=>c.groupLabel).map(c=>c.text)),[['样本数','样本率'],['好样本','坏样本','坏样本率'],[],[]])
})

test('单列表头跟随数据对齐，保留组标题及工作簿正文样式', () => {
  const page={first_row:11,rows:[[],[]],header_context:{first_row:4,last_row:5,cells:[
    cell(4,2,'模型性能',{bold:true,colspan:3,align:'center'}),
    cell(5,2,'统计项',{bold:true,align:'center'}),cell(5,3,'训练集',{bold:true}),cell(5,4,'测试集',{bold:true}),
  ]},cells:[
    cell(11,2,'AUC',{align:'left'}),cell(11,3,'58.4%',{align:'center',numeric:true}),cell(11,4,'56.5%',{numeric:true}),
    cell(12,2,'PSI',{align:'left'}),cell(12,3,'—',{align:'center'}),cell(12,4,'\\'),
  ]}
  const [table]=reportBlocks(page)
  assert.equal(table.rows[0].cells[0].align,'left')
  for(const row of table.rows.slice(1)) assert.deepEqual(row.cells.map(c=>c.align),['left','center','right'])
})
