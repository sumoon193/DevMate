<!--
  DocumentUpload 页面：Markdown / Plain Text / PDF / Office 入库。
-->
<script setup lang="ts">
import { onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getIngestionTask, uploadDocument } from '@/api/client'
import type { IngestionStatusResponse } from '@/types'

const tenantId = ref('tenant_001')
const departmentId = ref('dept_engineering')
const visibility = ref('department')
const uploading = ref(false)
const currentTask = ref<IngestionStatusResponse | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

function beforeUpload(file: File) {
  const allowed = ['.md', '.txt', '.pdf', '.docx', '.xlsx', '.pptx']
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase()
  if (!allowed.includes(ext)) {
    ElMessage.error(`仅支持 ${allowed.join(', ')} 格式`)
    return false
  }
  if (file.size > 10 * 1024 * 1024) {
    ElMessage.error('文件大小不能超过 10MB')
    return false
  }
  return true
}

async function handleUpload(option: { file: File }) {
  if (!beforeUpload(option.file)) return

  uploading.value = true
  currentTask.value = null

  try {
    const res = await uploadDocument(option.file, tenantId.value, departmentId.value, visibility.value)
    ElMessage.success(res.message)
    await startPolling(res.task_id)
  } catch (e: unknown) {
    ElMessage.error(`上传失败: ${(e as Error).message}`)
  } finally {
    uploading.value = false
  }
}

async function startPolling(taskId: string) {
  // 清除之前的轮询
  if (pollTimer) clearInterval(pollTimer)

  // 立即拉一次
  currentTask.value = await getIngestionTask(taskId)

  // 如果已经是终态，无需轮询
  if (isTerminal(currentTask.value?.status)) return

  // 每 2 秒轮询一次
  pollTimer = setInterval(async () => {
    try {
      currentTask.value = await getIngestionTask(taskId)
      if (isTerminal(currentTask.value?.status)) {
        clearInterval(pollTimer!)
        pollTimer = null
        if (currentTask.value?.status === 'ready') {
          ElMessage.success('入库完成')
        }
      }
    } catch {
      // 网络抖动不中断轮询
    }
  }, 2000)
}

function isTerminal(status: string | undefined): boolean {
  return status === 'ready' || status === 'failed'
}

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<template>
  <div class="document-page">
    <section class="surface upload-surface">
      <header class="page-section-header">
        <div>
          <p class="section-kicker">Document ingestion</p>
          <h2>把工程规范和运行手册写入证据层</h2>
        </div>
        <el-tag effect="plain">.md / .txt / .pdf / .docx / .xlsx / .pptx</el-tag>
      </header>

      <div class="document-grid">
        <el-form class="metadata-form" label-position="top">
          <el-form-item label="Tenant ID">
            <el-input v-model="tenantId" placeholder="tenant_001" />
          </el-form-item>
          <el-form-item label="Department ID">
            <el-input v-model="departmentId" placeholder="dept_engineering" />
          </el-form-item>
          <el-form-item label="Visibility">
            <el-segmented
              v-model="visibility"
              :options="[
                { label: '部门', value: 'department' },
                { label: '公开', value: 'public' },
                { label: '私有', value: 'private' },
                { label: '机密', value: 'confidential' },
              ]"
              block
            />
          </el-form-item>
        </el-form>

        <div class="drop-zone">
          <el-upload
            :http-request="handleUpload"
            :show-file-list="true"
            accept=".md,.txt,.pdf,.docx,.xlsx,.pptx"
            :limit="1"
            drag
            class="upload-area"
          >
            <el-icon :size="36"><UploadFilled /></el-icon>
            <div class="upload-text">
              拖入制度文件，或<em>选择文件</em>
            </div>
            <template #tip>
              <div class="upload-tip">单文件最大 10MB，入库后会写入 ACL metadata。</div>
            </template>
          </el-upload>
        </div>
      </div>
    </section>

    <aside class="surface status-surface">
      <header class="page-section-header compact">
        <div>
          <p class="section-kicker">Ingestion state</p>
          <h2>入库状态</h2>
        </div>
        <el-tag
          :type="currentTask?.status === 'ready' ? 'success' : currentTask?.status === 'failed' ? 'danger' : 'info'"
          effect="plain"
        >
          {{ currentTask?.status ?? 'idle' }}
        </el-tag>
      </header>

      <div v-if="currentTask" class="task-ledger" data-testid="ingestion-status">
        <div class="ledger-row">
          <span>Task ID</span>
          <strong class="mono">{{ currentTask.task_id }}</strong>
        </div>
        <div class="ledger-row">
          <span>Document ID</span>
          <strong class="mono">{{ currentTask.document_id }}</strong>
        </div>
        <div class="ledger-row">
          <span>Status</span>
          <strong>{{ currentTask.status }}</strong>
        </div>
        <div class="ledger-row">
          <span>Progress</span>
          <el-progress
            :percentage="Math.round(currentTask.progress * 100)"
            :status="currentTask.progress >= 1 ? 'success' : undefined"
          />
        </div>
        <el-alert
          v-if="currentTask.error_message"
          :title="currentTask.error_message"
          type="error"
          show-icon
        />
      </div>

      <div v-else class="empty-ledger">
        <el-icon><FolderChecked /></el-icon>
        <span>等待新的文档入库任务</span>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.document-page {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
  gap: 16px;
}

.upload-surface,
.status-surface {
  padding: 20px;
}

.page-section-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.page-section-header.compact {
  margin-bottom: 16px;
}

.section-kicker {
  margin-bottom: 5px;
  color: var(--color-muted);
  font-size: 12px;
}

.page-section-header h2 {
  color: var(--color-text);
  font-size: 19px;
  font-weight: 800;
  line-height: 1.3;
}

.document-grid {
  display: grid;
  grid-template-columns: minmax(260px, 360px) minmax(0, 1fr);
  gap: 20px;
}

.metadata-form {
  padding: 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-panel-muted);
}

.drop-zone {
  min-width: 0;
}

.upload-area {
  width: 100%;
}

.upload-area :deep(.el-upload-dragger) {
  display: grid;
  min-height: 240px;
  align-content: center;
  border-color: var(--color-border);
  border-radius: var(--radius);
  background: #ffffff;
}

.upload-area :deep(.el-upload-dragger:hover) {
  border-color: var(--color-primary);
}

.upload-text {
  margin-top: 10px;
  color: var(--color-text);
  font-weight: 700;
}

.upload-text em {
  color: var(--color-primary);
  font-style: normal;
}

.upload-tip {
  margin-top: 8px;
  color: var(--color-muted);
  font-size: 12px;
}

.task-ledger {
  display: grid;
  gap: 12px;
}

.ledger-row {
  display: grid;
  gap: 6px;
  padding: 12px;
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-sm);
  background: var(--color-panel-muted);
}

.ledger-row span {
  color: var(--color-muted);
  font-size: 12px;
}

.ledger-row strong {
  color: var(--color-text);
  font-size: 12px;
  word-break: break-all;
}

.empty-ledger {
  display: grid;
  place-items: center;
  min-height: 230px;
  gap: 10px;
  border: 1px dashed var(--color-border);
  border-radius: var(--radius);
  color: var(--color-muted);
}

.empty-ledger .el-icon {
  font-size: 34px;
  color: var(--color-primary);
}

@media (max-width: 1040px) {
  .document-page,
  .document-grid {
    grid-template-columns: 1fr;
  }
}
</style>
