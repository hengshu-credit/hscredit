import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
// 保留浏览器访问的 Host，使后端能按原始 Origin 检查同源请求。
const backend = {target:'http://127.0.0.1:8765', changeOrigin:false}
export default defineConfig({plugins:[vue()], build:{outDir:'../hscredit/resources/platform', emptyOutDir:true, chunkSizeWarningLimit:1600, rollupOptions:{output:{manualChunks:{charts:['echarts'], ui:['element-plus','@element-plus/icons-vue']}}}}, server:{proxy:{'/api':backend,'/services':backend}}})
