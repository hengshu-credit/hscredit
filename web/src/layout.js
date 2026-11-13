// 侧栏尺寸与天枢保持一致；内容区始终使用侧栏之外的全部宽度。
export const SIDEBAR_MIN_WIDTH=64
export const SIDEBAR_MAX_WIDTH=320
export const SIDEBAR_DEFAULT_WIDTH=220
export const SIDEBAR_COMPACT_THRESHOLD=168
export function clampSidebarWidth(value){
  return typeof value==='number'&&Number.isFinite(value)?Math.min(SIDEBAR_MAX_WIDTH,Math.max(SIDEBAR_MIN_WIDTH,Math.round(value))):SIDEBAR_DEFAULT_WIDTH
}
