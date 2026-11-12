let token=sessionStorage.getItem('hscredit-token')||''
if(location.hash.startsWith('#token=')){
  token=decodeURIComponent(location.hash.slice(7));sessionStorage.setItem('hscredit-token',token)
  history.replaceState(null,'',location.pathname)
}
export async function api(path, options={}){
  const headers={...(token?{Authorization:`Bearer ${token}`} : {}),...options.headers}
  if(options.body && !(options.body instanceof FormData))headers['Content-Type']='application/json'
  const response=await fetch('/api'+path,{...options,headers})
  if(!response.ok){let message;try{message=(await response.json()).detail}catch{message=`请求失败（${response.status}）`}throw new Error(typeof message==='string'?message:JSON.stringify(message))}
  return response.json()
}
export async function download(job,file){
  const blob=await artifactBlob(job,file)
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=file;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
}
export async function artifactBlob(job,file){
  return apiBlob(`/jobs/${job}/files/${encodeURIComponent(file)}`)
}
export async function apiBlob(path){
  const response=await fetch('/api'+path,{headers:token?{Authorization:`Bearer ${token}`}:{}})
  if(!response.ok)throw new Error('下载失败，请刷新任务状态后重试')
  return response.blob()
}
