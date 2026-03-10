import createClient from 'openapi-fetch';

import type { paths } from '@contracts/generated/schema';

function buildErrorMessage(fallback: string, response: Response, payload: unknown) {
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = String((payload as { detail?: unknown }).detail || '').trim();
    if (detail) {
      return detail;
    }
  }
  return `${fallback} (${response.status})`;
}

async function requireJsonResponse<T>(promise: Promise<{ data?: T; error?: unknown; response: Response }>, fallback: string): Promise<T> {
  const result = await promise;
  if (!result.response.ok || result.error || result.data == null) {
    throw new Error(buildErrorMessage(fallback, result.response, result.error));
  }
  return result.data;
}

export function getApiClient(apiBase: string) {
  return createClient<paths>({ baseUrl: apiBase });
}

export function fetchAppConfig(apiBase: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/config'), 'Could not load app config');
}

export function fetchSoccerNetConfig(apiBase: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/soccernet/config'), 'Could not load SoccerNet config');
}

export function fetchBackendJobs(apiBase: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/jobs'), 'Could not load backend jobs');
}

export function fetchRecentRuns(apiBase: string, limit = 1000, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/runs/recent', { params: { query: { limit } } }),
    'Could not load recent runs',
  );
}

export function fetchRun(apiBase: string, runId: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/runs/{run_id}', { params: { path: { run_id: runId } } }),
    'Could not load run',
  );
}

export function refreshRunDiagnostics(apiBase: string, runId: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).POST('/api/runs/{run_id}/refresh-diagnostics', { params: { path: { run_id: runId } } }),
    'Could not refresh diagnostics',
  );
}

export function fetchActiveExperiment(apiBase: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/experiments/active'),
    'Could not load active experiment',
  );
}
