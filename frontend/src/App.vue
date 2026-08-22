<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { healthCheck } from '@/api/client'
import type { HealthResponse } from '@/types'

const router = useRouter()
const route = useRoute()
const health = ref<HealthResponse | null>(null)
const healthFailed = ref(false)

const activePath = computed(() => (route.path.startsWith('/runs/') ? '/runs' : route.path))
const pageTitle = computed(() => {
  if (route.path.startsWith('/runs/')) return '运行审计详情'
  return ({
    '/': '研发事件工作台', '/runs': 'Agent Runs', '/documents': '知识库',
    '/memories': '长期记忆', '/eval': '评测中心', '/infrastructure': 'Trace 与基础设施',
  } as Record<string, string>)[route.path] ?? 'DevMate 控制台'
})
const apiModeLabel = computed(() => healthFailed.value ? 'API 不可用' : health.value?.mode === 'full' ? 'Full 模式' : 'Fallback 模式')
const envMeta = computed(() => {
  if (!health.value) return ['DevMate', '检查中']
  if (health.value.mode !== 'full') return ['仅离线验证', '未接入真实设施']
  const services = Object.values(health.value.services ?? {})
  return [`${services.filter(s => s.status === 'up').length}/${services.length} 服务在线`, health.value.status === 'ok' ? '就绪' : '降级']
})
const envDotClass = computed(() => ({ 'env-dot--down': healthFailed.value || health.value?.status !== 'ok' }))
onMounted(async()=>{try{health.value=await healthCheck()}catch{healthFailed.value=true}})
function handleSelect(path:string){if(path!==route.path)void router.push(path)}
</script>

<template>
  <div class="app-shell">
    <aside class="workspace-rail">
      <div class="brand-block"><div class="brand-mark"><el-icon><Cpu /></el-icon></div><div class="brand-copy"><span>Agent Runtime</span><strong>DevMate</strong></div></div>
      <el-menu :default-active="activePath" class="rail-menu" @select="handleSelect">
        <el-menu-item index="/"><el-icon><Monitor /></el-icon><span>事件工作台</span></el-menu-item>
        <el-menu-item index="/runs"><el-icon><Operation /></el-icon><span>Agent Runs</span></el-menu-item>
        <el-menu-item index="/documents"><el-icon><Document /></el-icon><span>知识库</span></el-menu-item>
        <el-menu-item index="/memories"><el-icon><Collection /></el-icon><span>长期记忆</span></el-menu-item>
        <el-menu-item index="/eval"><el-icon><DataAnalysis /></el-icon><span>评测中心</span></el-menu-item>
        <el-menu-item index="/infrastructure"><el-icon><Connection /></el-icon><span>Trace 与设施</span></el-menu-item>
      </el-menu>
      <div class="rail-footer"><p class="rail-label">当前环境</p><div class="env-line"><span class="env-dot" :class="envDotClass"/><span>{{apiModeLabel}}</span></div><div class="env-meta"><span>{{envMeta[0]}}</span><span>{{envMeta[1]}}</span></div></div>
    </aside>
    <section class="workspace">
      <header class="workspace-header"><div><p class="workspace-kicker">DevMate engineering operations</p><h1>{{pageTitle}}</h1></div><div class="status-strip" aria-label="系统边界"><div class="status-item"><span class="status-label">写操作</span><strong>审批门禁</strong></div><div class="status-item"><span class="status-label">回答依据</span><strong>引用可追溯</strong></div><div class="status-item"><span class="status-label">数据边界</span><strong>Tenant ACL</strong></div></div></header>
      <main class="app-main"><router-view/></main>
    </section>
  </div>
</template>
