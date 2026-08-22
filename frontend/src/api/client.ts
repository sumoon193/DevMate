/**
 * Typed API Client。
 *
 * 封装后端 FastAPI 8 个端点的调用，所有请求走 Vite proxy（/api → localhost:8000）。
 */
import type {
  HealthResponse,
  DocumentCreateResponse,
  IngestionStatusResponse,
  AgentRunCreateResponse,
  AgentRunDetail,
  ApprovalSubmitRequest,
  ApprovalSubmitResponse,
  EvalRunResponse,
  HRCase,
  CaseEventPage,
  RuntimeMetricsSnapshot,
  ApprovalDecisionType,
  MemoryPage,
  MemoryRecord,
  InfrastructureResponse,
} from '@/types'
import { accessToken } from '@/auth/session'

const BASE = '/api'

/** 通用 fetch wrapper，统一错误处理 */
async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const token = accessToken()
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}), ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json() as Promise<T>
}

// ── Health ─────────────────────────────────────────────────────────────

/** GET /health */
export function healthCheck(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

// ── Documents ──────────────────────────────────────────────────────────

/** POST /documents — multipart/form-data 文件上传 */
export async function uploadDocument(
  file: File,
  tenantId: string,
  departmentId: string,
  visibility: string = 'department',
): Promise<DocumentCreateResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('tenant_id', tenantId)
  form.append('department_id', departmentId)
  form.append('visibility', visibility)

  const token = accessToken()
  const res = await fetch(`${BASE}/documents`, {
    method: 'POST',
    body: form,
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json()
}

// ── Ingestion ──────────────────────────────────────────────────────────

/** GET /ingestions/{taskId} */
export function getIngestionTask(taskId: string): Promise<IngestionStatusResponse> {
  return request<IngestionStatusResponse>(`/ingestions/${taskId}`)
}

// ── Agent Runs ─────────────────────────────────────────────────────────

/** POST /agent-runs */
export function createAgentRun(query: string, userId: string): Promise<AgentRunCreateResponse> {
  return request<AgentRunCreateResponse>('/agent-runs', {
    method: 'POST',
    body: JSON.stringify({ query, user_id: userId }),
  })
}

/** GET /agent-runs/{runId} */
export function getAgentRun(runId: string): Promise<AgentRunDetail> {
  return request<AgentRunDetail>(`/agent-runs/${runId}`)
}

// ── Approvals ──────────────────────────────────────────────────────────

/** POST /agent-runs/{runId}/approvals/{approvalId} */
export function submitApproval(
  runId: string,
  approvalId: string,
  body: ApprovalSubmitRequest,
): Promise<ApprovalSubmitResponse> {
  return request<ApprovalSubmitResponse>(`/agent-runs/${runId}/approvals/${approvalId}`, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

// ── Eval ───────────────────────────────────────────────────────────────

/** POST /eval/runs */
export function runEval(datasetPath?: string): Promise<EvalRunResponse> {
  return request<EvalRunResponse>('/eval/runs', {
    method: 'POST',
    body: JSON.stringify({ dataset_path: datasetPath ?? null }),
  })
}

// ── Long-running Cases ──────────────────────────────────────────────

export function listCases(): Promise<HRCase[]> {
  return request<HRCase[]>('/cases')
}

export function getCase(caseId: string): Promise<HRCase> {
  return request<HRCase>(`/cases/${caseId}`)
}

export function createCase(): Promise<HRCase> {
  const suffix = Date.now().toString(36)
  return request<HRCase>('/cases', {
    method: 'POST',
    body: JSON.stringify({
      title: '新员工入职到转正',
      tenant_id: 'tenant_001',
      subject_user_id: `employee_${suffix}`,
      actor_id: 'user_hr',
      command_id: `cmd_case_create_${suffix}`,
    }),
  })
}

export function startCase(caseData: HRCase): Promise<HRCase> {
  return request<HRCase>(`/cases/${caseData.id}/start`, {
    method: 'POST',
    body: JSON.stringify({
      actor_id: 'user_hr',
      command_id: `cmd_case_start_${caseData.id}_${caseData.version}`,
      expected_version: caseData.version,
    }),
  })
}

export function decideCaseApproval(
  caseData: HRCase,
  approvalId: string,
  decision: ApprovalDecisionType,
): Promise<HRCase> {
  return request<HRCase>(`/cases/${caseData.id}/approvals/${approvalId}`, {
    method: 'POST',
    body: JSON.stringify({
      decision,
      actor_id: 'user_manager',
      command_id: `cmd_case_${decision}_${caseData.id}_${caseData.version}`,
      expected_version: caseData.version,
    }),
  })
}

export function getCaseEvents(caseId: string): Promise<CaseEventPage> {
  return request<CaseEventPage>(`/cases/${caseId}/events?after_sequence=0`)
}

export function refreshCasePolicy(
  caseData: HRCase,
  policyVersion: string,
): Promise<HRCase> {
  return request<HRCase>(`/cases/${caseData.id}/policies/refresh`, {
    method: 'POST',
    body: JSON.stringify({
      policy_version: policyVersion,
      actor_id: 'user_hr',
      command_id: `cmd_policy_refresh_${caseData.id}_${caseData.version}`,
      expected_version: caseData.version,
    }),
  })
}

export function getRuntimeMetrics(): Promise<RuntimeMetricsSnapshot> {
  return request<RuntimeMetricsSnapshot>('/metrics/runtime')
}

export function listMemories(tenantId: string): Promise<MemoryPage> {
  return request<MemoryPage>('/memories', { headers: { 'X-Tenant-ID': tenantId } })
}

export function deleteMemory(memoryId: string, tenantId: string): Promise<MemoryRecord> {
  return request<MemoryRecord>(`/memories/${memoryId}`, {
    method: 'DELETE',
    headers: { 'X-Tenant-ID': tenantId },
  })
}

export function getInfrastructure(): Promise<InfrastructureResponse> {
  return request<InfrastructureResponse>('/infrastructure')
}
