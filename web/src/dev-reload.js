// API 重载完成后重新挂载页面，重新读取模型、参数和数据目录。
let instance
let pending = false
let disposed = false

async function checkInstance() {
  if (pending) return
  pending = true
  try {
    const token = sessionStorage.getItem('hscredit-token')
    const response = await fetch('/api/health', {
      headers: token ? {Authorization: 'Bearer ' + token} : {},
      cache: 'no-store',
    })
    if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) return
    const data = await response.json()
    if (disposed || !data.instance) return
    // 首次连接也刷新，以恢复先启动前端时尚未加载的参数和数据。
    const previous = instance ?? sessionStorage.getItem('hscredit-dev-instance')
    instance = data.instance
    sessionStorage.setItem('hscredit-dev-instance', instance)
    if (previous !== instance) location.reload()
  } catch {
    // 服务重载期间可能短暂断开，下次检查继续连接。
  } finally {
    pending = false
  }
}

checkInstance()
const timer = setInterval(checkInstance, 1500)
if (import.meta.hot) import.meta.hot.dispose(() => {
  disposed = true
  clearInterval(timer)
})
