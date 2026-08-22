import { createRouter, createWebHistory } from 'vue-router'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: () => import('@/views/AgentConsole.vue') },
    { path: '/runs', component: () => import('@/views/AgentConsole.vue') },
    { path: '/documents', component: () => import('@/views/DocumentUpload.vue') },
    { path: '/memories', component: () => import('@/views/MemoryConsole.vue') },
    { path: '/eval', component: () => import('@/views/EvalResult.vue') },
    { path: '/infrastructure', component: () => import('@/views/InfrastructureView.vue') },
    { path: '/runs/:id', component: () => import('@/views/RunDetail.vue') },
  ],
})
