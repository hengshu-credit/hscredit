// 根据工作表的实际行列和合并范围，把本页内容排成章节、表格和图版。
export function reportBlocks(page) {
  const lines = new Map()
  function line(row) {
    if (!lines.has(row)) lines.set(row, {row, cells:[], occupied:new Map(), pictures:[]})
    return lines.get(row)
  }
  const cells = page.cells ?? page.rows.flatMap((row,i)=>row.flatMap((text,j)=>text===''?[]:[{
    text, row:page.first_row+i, col:j+1, rowspan:1, colspan:1, bold:false,
  }]))
  for (const cell of [...(page.header_context?.cells || []), ...cells]) {
    line(cell.row).cells.push(cell)
    for (let r=cell.row; r<cell.row+cell.rowspan; r++) {
      for (let c=cell.col; c<cell.col+cell.colspan; c++) line(r).occupied.set(c,cell)
    }
  }
  for (const picture of page.pictures || []) line(picture.row).pictures.push(picture)
  const blocks = []
  let group = [], pictures = []
  function table(left, right) {
    const widths = Array(right-left+1).fill(64)
    const rows = group.map(l => {
      const cells = []
      for (let col=left; col<=right; col++) {
        const cell = l.occupied.get(col)
        if (!cell) {
          cells.push({col, text:'', rowspan:1, colspan:1})
          continue
        }
        if (cell.row!==l.row || cell.col!==col) continue
        cells.push(cell)
        if (cell.colspan===1) {
          const units = (cell.display ?? cell.text).slice(0,100).split('\n').map(s=>s.replace(/[^\x00-\x7f]/g,'xx').length)
          widths[col-left] = Math.max(widths[col-left], Math.min(260,22 + Math.max(...units)*7))
        }
      }
      const values = cells.filter(c=>c.text!=='')
      const header = values.length>0 && values.every(c=>c.bold)
      return {row:l.row, cells, header}
    })
    // 旧版报告可能将不同父指标下的同名子标题合为一个单元格。
    for (let i=1; i<rows.length; i++) {
      if (!rows[i].header || !rows[i-1].header) continue
      const parentStarts = rows[i-1].cells.filter(c=>c.bold).map(c=>c.col)
      rows[i].cells = rows[i].cells.flatMap(cell => {
        if (cell.colspan<=1) return [cell]
        const end = cell.col+cell.colspan
        const boundaries = [cell.col, ...parentStarts.filter(col=>cell.col<col && col<end), end]
        return boundaries.slice(0,-1).map((col,j)=>({...cell,col,colspan:boundaries[j+1]-col}))
      })
    }
    // 叶表头沿用本列数据的对齐方式。跨列标题保留工作簿的分组或标题样式。
    const alignments = new Map()
    for (const row of rows) {
      if (row.header) continue
      for (const cell of row.cells) {
        if (cell.colspan!==1 || cell.text==='') continue
        // PSI 等指标用反斜杠或横线表示无基准值，不据此把整列判为文本。
        if (!cell.align && !cell.numeric && ['\\','—','–','-'].includes(cell.text)) continue
        if (!alignments.has(cell.col)) alignments.set(cell.col,new Set())
        alignments.get(cell.col).add(cell.align ?? (cell.numeric?'right':'left'))
      }
    }
    for (const row of rows) {
      row.cells = row.cells.map(cell => {
        const column = alignments.get(cell.col)
        if (cell.colspan!==1 || !column || (!row.header && cell.align)) return cell
        const align = column.size===1 ? column.values().next().value : (cell.align ?? 'left')
        return cell.align===align ? cell : {...cell,align}
      })
    }
    return {type:'table', row:rows[0].row, left, rows, widths, minWidth:widths.reduce((a,b)=>a+b,0)}
  }

  function flush() {
    if (group.length) {
      // 整段均为空的列才是两张表之间的间隔；合并表头和数据跨行都算占用。
      const columns = [...new Set(group.flatMap(l=>[...l.occupied.keys()]))].sort((a,b)=>a-b)
      const ranges = []
      for (const col of columns) {
        const range = ranges.at(-1)
        if (range && col===range[1]+1) range[1]=col
        else ranges.push([col,col])
      }
      const tables = ranges.map(([left,right])=>table(left,right))
      const first = tables[0]
      if (tables.length>1) blocks.push({type:'parallel',row:first.row,tables})
      else if (first.rows.length===1 && first.rows[0].cells.length===1 && !first.rows[0].cells[0].numeric) {
        blocks.push({type:'paragraph',row:first.row,cell:first.rows[0].cells[0]})
      } else blocks.push(first)
    }
    if (pictures.length) blocks.push({type:'pictures', row:pictures[0].row, pictures})
    group = []
    pictures = []
  }
  if (page.header_context) {
    for (let r=page.header_context.first_row; r<=page.header_context.last_row; r++) {
      if (lines.get(r)?.occupied.size) group.push(lines.get(r))
    }
  }
  for (let r=page.first_row; r<page.first_row+page.rows.length; r++) {
    const l = lines.get(r)
    if (!l) {
      // 并列小标题后的一个空行是 Excel 的留白，预览中标题仍归属各自的表格。
      const captions=group.length===1 && group[0].cells
      if (captions && captions.length>1 && captions.every(c=>c.bold && c.colspan>1 && lines.get(r+1)?.occupied.get(c.col)?.bold)) continue
      flush()
      continue
    }
    const single = l.cells.length===1 ? l.cells[0] : null
    const next = lines.get(r+1)
    const numbered = single && /^([一二三四五六七八九十]+|\d+(?:\.\d+)*)(?:、|[.．]\s)/.test(single.text)
    const groupHeader = !numbered && single?.colspan>1 && next?.cells.some(c=>c.bold && c.col>=single.col && c.col<single.col+single.colspan)
    if (single?.bold && single.heading!==false && !groupHeader && !single.numeric && single.rowspan===1 && l.occupied.size===single.colspan) {
      flush()
      blocks.push({type:'heading', row:r, cell:single})
    } else if (l.occupied.size) group.push(l)
    if (l.pictures.length) {
      pictures.push(...l.pictures.sort((a,b)=>a.col-b.col))
      // 图片和表格共行时保留跨行表头，等该合并范围结束后展示图片。
      if (!l.occupied.size) flush()
    }
  }
  flush()
  if (!page.pictures && page.image_count) {
    blocks.push({type:'pictures', row:page.first_row+page.rows.length,
      pictures:Array.from({length:page.image_count},(_,index)=>({index}))})
  }
  return blocks
}
