<!--
  AgentConsole 页面：DevMate 研发事件与 Agent Run 工作台。
-->
<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useAgentStore } from '@/stores/agent'
import ApprovalCard from '@/components/ApprovalCard.vue'
import CitationPanel from '@/components/CitationPanel.vue'
import StepTimeline from '@/components/StepTimeline.vue'
import type { ChatMessage } from '@/types'

const agentStore = useAgentStore()

const query = ref('')
const userId = ref('user_001')
const messages = ref<ChatMessage[]>([])
const chatContainer = ref<HTMLElement | null>(null)
const activeTab = ref('steps')

const exampleQueries = [
  '分析最近一次部署失败，并列出证据与恢复步骤',
  '检索知识库中的数据库迁移规范',
  '为这个高风险修复生成审批计划',
]

const citationCount = computed(() => {
  const citations = agentStore.currentRun?.result?.citations
  return Array.isArray(citations) ? citations.length : 0
})

const runStatusLabel = computed(() => agentStore.currentRun?.status ?? 'idle')

function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

async function handleSubmit() {
  const q = query.value.trim()
  if (!q) return

  messages.value.push({
    id: `msg_${Date.now()}`,
    role: 'user',
    content: q,
    timestamp: new Date(),
  })
  query.value = ''
  scrollToBottom()

  try {
    const runId = await agentStore.startRun(q, userId.value)

    messages.value.push({
      id: `msg_${Date.now()}`,
      role: 'assistant',
      content: '',
      runId,
      timestamp: new Date(),
    })
    scrollToBottom()
  } catch (e: unknown) {
    ElMessage.error(`创建 Run 失败: ${(e as Error).message}`)
  }
}

async function handleApproval(
  runId: string,
  approvalId: string,
  decision: 'approve' | 'edit' | 'reject',
  editedParams?: Record<string, unknown>,
) {
  try {
    await agentStore.approve(runId, approvalId, decision, editedParams)
    ElMessage.success(`审批已提交: ${decision}`)
    scrollToBottom()
  } catch (e: unknown) {
    ElMessage.error(`审批失败: ${(e as Error).message}`)
  }
}

function statusType(status?: string) {
  if (status === 'completed') return 'success'
  if (status === 'failed' || status === 'cancelled') return 'danger'
  if (status === 'awaiting_approval') return 'warning'
  return 'info'
}

function answerText(result: Record<string, unknown> | null): string {
  if (!result) return '暂无结果'
  const answer = result.answer
  if (typeof answer === 'string' && answer.trim()) return answer
  return JSON.stringify(result, null, 2)
}

function confidenceText(result: Record<string, unknown> | null): string | null {
  if (!result || typeof result.confidence !== 'number') return null
  return `${Math.round(result.confidence * 100)}%`
}
</script>

<template>
  <div class="agent-console">
    <section class="run-strip">
      <div class="run-strip-item">
        <span>Run 状态</span>
        <strong>{{ runStatusLabel }}</strong>
      </div>
      <div class="run-strip-item">
        <span>待审批</span>
        <strong>{{ agentStore.pendingApprovals.length }}</strong>
      </div>
      <div class="run-strip-item">
        <span>引用证据</span>
        <strong>{{ citationCount }}</strong>
      </div>
      <div class="run-strip-item">
        <span>SSE 事件</span>
        <strong>{{ agentStore.sseEvents.length }}</strong>
      </div>
    </section>

    <div class="console-grid">
      <section class="surface chat-panel">
        <header class="panel-header">
          <div>
            <p class="panel-eyebrow">Engineering incident runtime</p>
            <h2>证据、计划、工具和审批在同一条 Run 中闭环</h2>
          </div>
          <el-tag :type="statusType(agentStore.currentRun?.status)" effect="plain">
            {{ runStatusLabel }}
          </el-tag>
        </header>

        <div ref="chatContainer" class="chat-messages">
          <div v-if="messages.length === 0" class="ready-state">
            <div class="ready-header">
              <el-icon><Tickets /></el-icon>
              <div>
                <strong>运行台就绪</strong>
                <span>先创建 Agent Run，再检查 evidence、plan、approval 和 trace。</span>
              </div>
            </div>
            <div class="example-queries">
              <button
                v-for="example in exampleQueries"
                :key="example"
                type="button"
                @click="query = example"
              >
                {{ example }}
              </button>
            </div>
          </div>

          <div
            v-for="msg in messages"
            :key="msg.id"
            :class="['message', msg.role]"
          >
            <div class="message-meta">
              <span>{{ msg.role === 'user' ? 'User' : 'Harness' }}</span>
              <time>{{ msg.timestamp.toLocaleTimeString('zh-CN') }}</time>
            </div>
            <div class="message-bubble">
              <template v-if="msg.role === 'user'">
                <p>{{ msg.content }}</p>
              </template>

              <template v-else>
                <div v-if="agentStore.loading" class="loading-line">
                  <span />
                  <span />
                  <span />
                  <em>正在创建运行记录</em>
                </div>

                <template v-else-if="agentStore.currentRun">
                  <div class="run-inline-status">
                    <el-tag :type="statusType(agentStore.currentRun.status)" size="small">
                      {{ agentStore.currentRun.status }}
                    </el-tag>
                    <span class="mono">{{ agentStore.currentRun.id }}</span>
                  </div>

                  <ApprovalCard
                    v-for="approval in agentStore.pendingApprovals"
                    :key="approval.id"
                    :approval="approval"
                    :run-id="agentStore.currentRun.id"
                    @decide="(d, p) => handleApproval(agentStore.currentRun!.id, approval.id, d, p)"
                  />

                  <div v-if="agentStore.currentRun.result" class="answer-section">
                    <div class="answer-header">
                      <span>最终答复</span>
                      <el-tag
                        v-if="confidenceText(agentStore.currentRun.result)"
                        size="small"
                        effect="plain"
                      >
                        置信度 {{ confidenceText(agentStore.currentRun.result) }}
                      </el-tag>
                    </div>
                    <p class="answer-text" data-testid="answer-text">
                      {{ answerText(agentStore.currentRun.result) }}
                    </p>
                  </div>
                </template>

                <el-alert
                  v-if="agentStore.error"
                  :title="agentStore.error"
                  type="error"
                  show-icon
                  :closable="false"
                />
              </template>
            </div>
          </div>
        </div>

        <footer class="command-bar">
          <el-input
            v-model="query"
            placeholder="输入研发事件、知识检索或修复任务"
            :disabled="agentStore.loading"
            size="large"
            @keyup.enter="handleSubmit"
          >
            <template #prepend>
              <el-input v-model="userId" class="user-id-input" />
            </template>
            <template #append>
              <el-button
                type="primary"
                :loading="agentStore.loading"
                @click="handleSubmit"
              >
                <el-icon><Promotion /></el-icon>
                创建 Run
              </el-button>
            </template>
          </el-input>
        </footer>
      </section>

      <aside class="surface detail-panel">
        <el-tabs v-model="activeTab" type="border-card" stretch>
          <el-tab-pane name="steps">
            <template #label>
              <span class="tab-label">
                <el-icon><List /></el-icon>
                Trace
              </span>
            </template>
            <StepTimeline
              :steps="agentStore.currentRun?.steps ?? []"
              :sse-events="agentStore.sseEvents"
            />
          </el-tab-pane>

          <el-tab-pane name="citations">
            <template #label>
              <span class="tab-label">
                <el-icon><DocumentCopy /></el-icon>
                Evidence
              </span>
            </template>
            <CitationPanel
              :citations="(agentStore.currentRun?.result?.citations as any[]) ?? []"
            />
          </el-tab-pane>

          <el-tab-pane name="tools">
            <template #label>
              <span class="tab-label">
                <el-icon><Connection /></el-icon>
                Tools
              </span>
            </template>
            <div
              v-if="agentStore.currentRun?.tool_calls?.length"
              class="tool-list"
              data-testid="tool-list"
            >
              <article
                v-for="(tc, idx) in agentStore.currentRun.tool_calls"
                :key="idx"
                class="operation-card"
              >
                <div class="operation-head">
                  <strong>{{ tc.tool_name }}</strong>
                  <el-tag
                    v-if="tc.risk_level"
                    :type="tc.risk_level === 'write' ? 'warning' : 'info'"
                    size="small"
                    effect="plain"
                  >
                    {{ tc.risk_level }}
                  </el-tag>
                </div>
                <pre class="json-block">{{ JSON.stringify(tc.parameters ?? {}, null, 2) }}</pre>
                <pre v-if="tc.result" class="json-block result-block">{{ JSON.stringify(tc.result, null, 2) }}</pre>
              </article>
            </div>
            <el-empty v-else description="暂无工具调用" :image-size="60" />
          </el-tab-pane>

          <el-tab-pane name="approvals">
            <template #label>
              <span class="tab-label">
                <el-icon><WarningFilled /></el-icon>
                Approval
              </span>
            </template>
            <div
              v-if="agentStore.currentRun?.approvals?.length"
              class="approval-list"
              data-testid="approval-list"
            >
              <article
                v-for="approval in agentStore.currentRun.approvals"
                :key="approval.id"
                class="approval-record"
              >
                <div>
                  <strong>{{ approval.tool_name }}</strong>
                  <span class="mono">{{ approval.id }}</span>
                </div>
                <el-tag
                  :type="approval.status === 'approved' ? 'success' : approval.status === 'rejected' ? 'danger' : 'warning'"
                  size="small"
                >
                  {{ approval.decision ?? approval.status }}
                </el-tag>
              </article>
            </div>
            <el-empty v-else description="暂无审批记录" :image-size="60" />
          </el-tab-pane>
        </el-tabs>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.agent-console {
  display: grid;
  gap: 16px;
}

.run-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.run-strip-item {
  padding: 13px 14px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: rgba(255, 255, 255, 0.86);
}

.run-strip-item span {
  display: block;
  color: var(--color-muted);
  font-size: 12px;
}

.run-strip-item strong {
  display: block;
  margin-top: 4px;
  color: var(--color-text);
  font-size: 20px;
  line-height: 1.1;
}

.console-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(360px, 440px);
  gap: 16px;
  min-height: calc(100vh - 224px);
}

.chat-panel {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex-direction: column;
  overflow: hidden;
}

.panel-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 20px;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-panel);
}

.panel-eyebrow {
  margin-bottom: 4px;
  color: var(--color-muted);
  font-size: 12px;
}

.panel-header h2 {
  color: var(--color-text);
  font-size: 18px;
  font-weight: 800;
  line-height: 1.35;
}

.chat-messages {
  flex: 1;
  min-height: 360px;
  overflow-y: auto;
  padding: 18px 20px;
  background:
    linear-gradient(180deg, #fbfcfd 0%, #ffffff 48%, #f8fafb 100%);
}

.ready-state {
  display: grid;
  gap: 18px;
  max-width: 680px;
  padding: 18px;
  border: 1px dashed var(--color-border);
  border-radius: var(--radius);
  background: rgba(255, 255, 255, 0.74);
}

.ready-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.ready-header .el-icon {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  border-radius: var(--radius-sm);
  color: #ffffff;
  background: var(--color-primary);
}

.ready-header strong,
.ready-header span {
  display: block;
}

.ready-header strong {
  color: var(--color-text);
  font-size: 16px;
}

.ready-header span {
  margin-top: 3px;
  color: var(--color-muted);
  font-size: 13px;
}

.example-queries {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.example-queries button {
  max-width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-blue);
  background: #ffffff;
  cursor: pointer;
}

.example-queries button:hover {
  border-color: var(--color-blue);
}

.message {
  display: grid;
  gap: 6px;
  margin-bottom: 16px;
}

.message.user {
  justify-items: end;
}

.message.assistant {
  justify-items: start;
}

.message-meta {
  display: flex;
  gap: 8px;
  color: var(--color-subtle);
  font-size: 11px;
}

.message-bubble {
  width: fit-content;
  max-width: min(780px, 88%);
  padding: 12px 14px;
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius);
  color: var(--color-text);
  background: #ffffff;
  font-size: 14px;
  line-height: 1.65;
  box-shadow: 0 4px 16px rgba(25, 39, 54, 0.05);
}

.message.user .message-bubble {
  border-color: #cddbe8;
  color: #13283f;
  background: #eef5fb;
}

.run-inline-status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.run-inline-status .mono {
  color: var(--color-muted);
  font-size: 12px;
}

.answer-section {
  display: grid;
  gap: 8px;
  margin-top: 10px;
  padding: 12px;
  border: 1px solid #cbded6;
  border-radius: var(--radius-sm);
  background: var(--color-green-soft);
}

.answer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--color-primary-strong);
  font-size: 13px;
  font-weight: 800;
}

.answer-text {
  white-space: pre-wrap;
  color: #1f342d;
}

.loading-line {
  display: flex;
  align-items: center;
  gap: 5px;
  color: var(--color-muted);
}

.loading-line span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-primary);
  animation: pulse 1.2s infinite ease-in-out;
}

.loading-line span:nth-child(2) {
  animation-delay: 0.15s;
}

.loading-line span:nth-child(3) {
  animation-delay: 0.3s;
}

.loading-line em {
  margin-left: 6px;
  font-style: normal;
  font-size: 12px;
}

@keyframes pulse {
  0%,
  80%,
  100% {
    opacity: 0.28;
    transform: translateY(0);
  }

  40% {
    opacity: 1;
    transform: translateY(-2px);
  }
}

.command-bar {
  padding: 14px 16px;
  border-top: 1px solid var(--color-border);
  background: #ffffff;
}

.user-id-input {
  width: 128px;
}

.detail-panel {
  min-width: 0;
  overflow: hidden;
}

.detail-panel :deep(.el-tabs) {
  height: 100%;
}

.detail-panel :deep(.el-tabs__content) {
  height: calc(100% - 48px);
  overflow-y: auto;
  padding: 14px;
}

.tab-label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.tool-list,
.approval-list {
  display: grid;
  gap: 10px;
}

.operation-card,
.approval-record {
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: #ffffff;
}

.operation-head,
.approval-record {
  align-items: center;
  justify-content: space-between;
}

.operation-head {
  display: flex;
  gap: 10px;
}

.approval-record {
  grid-template-columns: minmax(0, 1fr) auto;
}

.approval-record strong,
.approval-record span {
  display: block;
}

.approval-record .mono {
  margin-top: 3px;
  color: var(--color-subtle);
  font-size: 11px;
}

.result-block {
  border-color: #cbded6;
  background: var(--color-green-soft);
}

@media (max-width: 1180px) {
  .console-grid {
    grid-template-columns: 1fr;
  }

  .detail-panel {
    min-height: 480px;
  }
}

@media (max-width: 760px) {
  .run-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .panel-header,
  .answer-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .message-bubble {
    max-width: 100%;
  }

  .command-bar :deep(.el-input-group__prepend) {
    display: none;
  }
}
</style>
