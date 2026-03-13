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

export function fetchBenchmarkConfig(apiBase: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/benchmark/config'),
    'Could not load benchmark config',
  );
}

export function fetchBenchmarkHistory(apiBase: string, limit = 20, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/benchmark/history', { params: { query: { limit } } }),
    'Could not load benchmark history',
  );
}

export function fetchBenchmarkJob(apiBase: string, benchmarkId: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).GET('/api/benchmark/jobs/{benchmark_id}', { params: { path: { benchmark_id: benchmarkId } } }),
    'Could not load benchmark job',
  );
}

export function runBenchmark(
  apiBase: string,
  payload: { suite_ids: string[]; recipe_ids: string[]; label?: string },
  fetchImpl?: typeof fetch,
) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).POST('/api/benchmark/run', {
      body: {
        suite_ids: payload.suite_ids,
        recipe_ids: payload.recipe_ids,
        label: payload.label ?? '',
      },
    }),
    'Could not start benchmark',
  );
}

export function ensureBenchmarkClip(apiBase: string, sourcePath: string, fetchImpl?: typeof fetch) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).POST('/api/benchmark/ensure-clip', { body: { source_path: sourcePath } }),
    'Could not prepare benchmark clip',
  );
}

export function importBenchmarkLocal(
  apiBase: string,
  payload: { checkpoint_path: string; label?: string },
  fetchImpl?: typeof fetch,
) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).POST('/api/benchmark/candidates/import-local', {
      body: {
        checkpoint_path: payload.checkpoint_path,
        label: payload.label ?? '',
      },
    }),
    'Could not import local checkpoint',
  );
}

export function importBenchmarkHf(
  apiBase: string,
  payload: { repo_id: string; filename?: string; label?: string },
  fetchImpl?: typeof fetch,
) {
  return requireJsonResponse(
    createClient<paths>({ baseUrl: apiBase, fetch: fetchImpl }).POST('/api/benchmark/candidates/import-hf', {
      body: {
        repo_id: payload.repo_id,
        filename: payload.filename ?? 'best.pt',
        label: payload.label ?? '',
      },
    }),
    'Could not import Hugging Face checkpoint',
  );
}
