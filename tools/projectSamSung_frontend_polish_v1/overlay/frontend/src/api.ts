/** FastAPI V1.1 contract. Only relative /api paths: Vite proxies to local port 8000. */
export type M1Mode = 'skip' | 'presegmented' | 'pyvi_experimental';

export interface ArticleForm {
  title: string;
  description: string;
  articleText: string;
  articleSegmented: string;
  m1Mode: M1Mode;
}

export interface Prediction {
  label: string;
  prediction: number;
  threshold: number;
  probability_error?: number;
  probability_clickbait?: number;
  probability_consistent?: number;
  chunk_count?: number;
  branches?: Record<string, unknown>;
  warnings?: string[];
  [key: string]: unknown;
}

export interface ModuleResponse {
  status: 'completed' | 'skipped' | string;
  result?: Prediction;
  input_mode?: string;
  warning?: string;
  reason?: string;
}

export interface AnalyzeResponse {
  api_version: string;
  module1_spelling: ModuleResponse;
  module2_clickbait: ModuleResponse;
  module3_consistency: ModuleResponse;
  notes?: string[];
}

export interface HealthResponse {
  status: string;
  models_ready: boolean;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch('/api/v1/health', { cache: 'no-store', signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<HealthResponse>;
}

const fieldNames: Record<string, string> = {
  title: 'Tiêu đề', description: 'Mô tả', article_text: 'Nội dung bài báo',
  article_segmented: 'Văn bản đã phân đoạn', m1_input_mode: 'Chế độ M1',
};

function parseApiError(status: number, value: unknown): string {
  if (status === 503) return 'Máy chủ chưa sẵn sàng hoặc đang bận. Hãy thử lại sau ít phút.';
  if (status === 500) return 'Có lỗi trong quá trình suy luận. Hãy xem cửa sổ CMD chạy FastAPI.';
  if (status === 429) return 'Có quá nhiều yêu cầu. Hãy đợi rồi thử lại.';
  if (status === 502 || status === 504) return 'Máy chủ không phản hồi kịp thời. Kiểm tra FastAPI và thử lại.';
  if (status === 422) {
    if (value && typeof value === 'object' && 'detail' in value) {
      const detail = (value as {detail: unknown}).detail;
      if (typeof detail === 'string') {
        if (/article_segmented.*or.*m1_input_mode/i.test(detail))
          return 'Chỉ chọn một cách nhập M1: văn bản đã phân đoạn hoặc PyVi thử nghiệm.';
        if (/article_segmented/i.test(detail)) return 'Văn bản đã phân đoạn không hợp lệ hoặc đang để trống.';
        if (/article_text/i.test(detail)) return 'Nội dung bài báo không hợp lệ hoặc đang để trống.';
        if (/title/i.test(detail)) return 'Tiêu đề bài báo không hợp lệ hoặc đang để trống.';
      }
      if (Array.isArray(detail)) {
        const issues = detail.map((item: unknown) => {
          if (!item || typeof item !== 'object') return 'Dữ liệu đầu vào không hợp lệ';
          const entry = item as {loc?: unknown};
          const lastField = Array.isArray(entry.loc) ? String(entry.loc.at(-1)) : '';
          return `${fieldNames[lastField] ?? 'Trường đầu vào'} không hợp lệ`;
        });
        return [...new Set(issues)].join(' · ');
      }
    }
    return 'Dữ liệu đầu vào không hợp lệ. Hãy kiểm tra các trường và thử lại.';
  }
  return `Yêu cầu thất bại (HTTP ${status}). Hãy kiểm tra kết nối FastAPI.`;
}

export async function analyze(form: ArticleForm, signal?: AbortSignal): Promise<AnalyzeResponse> {
  const body: Record<string, string> = {
    title: form.title.trim(),
    description: form.description.trim(),
    article_text: form.articleText.trim(),
  };
  if (form.m1Mode === 'presegmented') body.article_segmented = form.articleSegmented.trim();
  if (form.m1Mode === 'pyvi_experimental') body.m1_input_mode = 'pyvi_experimental';
  // In "skip" mode neither optional field is sent, leaving M1 skipped by the API.
  let response: Response;
  try {
    response = await fetch('/api/v1/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new Error('Không kết nối được với FastAPI tại http://127.0.0.1:8000.');
  }
  if (!response.ok) {
    const details: unknown = await response.json().catch(() => null);
    throw new Error(parseApiError(response.status, details));
  }
  return response.json() as Promise<AnalyzeResponse>;
}
